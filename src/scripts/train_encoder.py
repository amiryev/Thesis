import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from pathlib import Path
import time
import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import numpy as np

from src.utils import config
from src.core.encoder import XrayEncoder
from src.data.dataset import MultiPatientDRRDataset, DRRMetadataDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices

def parse_args():
    """
    Parses command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train XrayEncoder (Masked Reconstruction)")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")

    # Encoder Specific
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Ratio of patches to mask")
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE, help="Size of patches")
    
    # Paths
    parser.add_argument("--ckpt_dir", type=Path, default=Path(config.CKPT_DIR), help="Directory to save checkpoints/logs")
    parser.add_argument("--data_dir", type=Path, default=Path(config.DATA_DIR), help="Root directory containing CT/ and CRM/")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save test results")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")

    # Testing mode
    parser.add_argument("--on_the_fly", action="store_true", help="Generate DRR while training or use pre-generated dataset")
    parser.add_argument("--ddp", action="store_false", help="Use DDP parallelization")
    parser.add_argument("--test", action="store_true", help="Run in test mode to generate visuals and evaluate features")

    return parser.parse_args()


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes L1 loss only on masked pixels.
    
    Args:
        pred (torch.Tensor): (B, 1, H, W) Reconstructed image.
        target (torch.Tensor): (B, 1, H, W) Original image.
        pixel_mask (torch.Tensor): (B, 1, H, W) Binary mask (1 where generated/masked, 0 where kept).
        
    Returns:
        torch.Tensor: Scaled L1 loss for the masked areas.
    """
    diff = torch.abs(pred - target) * pixel_mask
    # Normalize by number of masked pixels to avoid scaling issues with mask ratio
    denom = pixel_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


class EncoderTrainer:
    """
    Trainer class encapsulating the training logic for the XRay Encoder.
    """
    def __init__(self, rank: int, world_size: int, args: argparse.Namespace):
        """
        Initializes the trainer with distributed settings, models, and data loaders.
        
        Args:
            rank (int): Current process rank.
            world_size (int): Total number of distributed processes.
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = torch.device("cuda", rank)
        
        # Setup Distributed Environment
        if args.ddp:
            DDPHelper.setup(rank, world_size)

        seed = 42
        torch.manual_seed(seed + rank)
        
        # Logging setup (Rank 0 only)
        self.logger = None
        if self.rank == 0:
            self.args.ckpt_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logger = setup_logger("train_encoder", self.args.ckpt_dir / f"train_{timestamp}.log")
            self.logger.info(f"Starting Encoder Training on {world_size} GPUs")
            self.logger.info(f"Arguments: {vars(args)}")
            
        self._setup_data()
        self._setup_model()
        
        self.ckpt_manager = CheckpointManager(self.args.ckpt_dir, self.rank)
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'loss': []}
        
        self._resume_checkpoint()

    def _setup_data(self):
        """Builds the dataset and standard/distributed dataloaders."""
        data_dir = self.args.data_dir
        if not data_dir.exists():
            if self.rank == 0 and self.logger:
                self.logger.error(f"Data directory not found at {data_dir}")
            if self.args.ddp:
                DDPHelper.cleanup()
            sys.exit(1)
            
        if self.args.on_the_fly:
            self.dataset = MultiPatientDRRDataset(
                data_dir=data_dir,
                device='cpu',
                size=config.IMAGE_SIZE,
                patient_ids=(7, 11)
            )
        else:
            self.dataset = DRRMetadataDataset(root_dir=data_dir)

        self.sampler = DistributedSampler(
            self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        self.loader = DataLoader(
            self.dataset, 
            batch_size=self.args.batch_size, 
            sampler=self.sampler, 
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def _setup_model(self):
        """Initializes the encoder model, wraps it in DDP, and sets up the optimizer."""
        model = XrayEncoder(
            device=self.device, 
            size=config.IMAGE_SIZE, 
            patch_size=self.args.patch_size
        ).to(self.device)
        
        if self.args.ddp:
            self.model = DDP(model, device_ids=[self.rank], find_unused_parameters=False)
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )

    def _resume_checkpoint(self):
        """Loads model and optimizer states if a resume checkpoint path is provided."""
        if self.args.resume:
            ckpt = self.ckpt_manager.load(self.args.resume, self.device)
            if ckpt:
                self.model.load_state_dict(ckpt['model_sd'])
                self.optimizer.load_state_dict(ckpt['optim_sd'])
                self.start_epoch = ckpt['epoch'] + 1
                self.best_loss = ckpt.get('best_loss', float('inf'))
                self.history = ckpt.get('history', {'loss': []})
                if self.rank == 0 and self.logger:
                    self.logger.info(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.5f}")
            else:
                if self.rank == 0 and self.logger:
                    self.logger.warning(f"Checkpoint {self.args.resume} not found, starting from scratch.")

    def plot_loss_curve(self, save_path: Path):
        """
        Plots the training loss curve and saves it to a file.
        
        Args:
            save_path (Path): Path where the plot image will be saved.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.history['loss']) + 1), self.history['loss'], label='Train Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def train_one_epoch(self, epoch: int):
        """
        Executes a single epoch of training.
        
        Args:
            epoch (int): The current epoch index.
            
        Returns:
            tuple: (average_loss, original_imgs, masks, reconstructed_imgs) from the last batch.
        """
        self.model.train()
        meters = AverageMeter()
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
        else:
            pbar = self.loader

        for i, batch in enumerate(pbar):
            imgs = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            
            # Forward pass provides reconstruction, mask, and latent representations
            recon, pixel_mask, _ = self.model(imgs, mask_ratio=self.args.mask_ratio)
            
            # Loss calculation against the masked pixels
            loss = masked_l1_loss(recon, imgs, pixel_mask)
            loss.backward()
            self.optimizer.step()
            
            meters.update(loss.item(), imgs.size(0))
            if self.rank == 0:
                pbar.set_postfix(loss=f"{meters.avg:.5f}")

        return meters.avg

    def train(self):
        """
        Main training loop handling epochs, logging, and checkpointing.
        """
        if self.rank == 0 and self.logger:
            self.logger.info("Training loop started...")
        
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.sampler.set_epoch(epoch)
            avg_loss = self.train_one_epoch(epoch)
            
            if self.rank == 0:
                self.history['loss'].append(avg_loss)
                self.plot_loss_curve(self.args.ckpt_dir / "loss_curve.png")
                
                elapsed = time.time() - start_time
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} | Loss: {avg_loss:.6f} | Elapsed: {elapsed:.1f}s")
                
                # Checkpoints state dictionary
                state = {
                    'epoch': epoch,
                    'model_sd': self.model.state_dict(),
                    'optim_sd': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'history': self.history,
                    'config': vars(self.args)
                }
                
                # Always save last checkpoint
                self.ckpt_manager.save("encoder_last", state)
                
                # Save best checkpoint and visualization
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    state['best_loss'] = self.best_loss
                    self.ckpt_manager.save("encoder_best", state)
                    self.logger.info(f"New best model saved with loss {self.best_loss:.6f}")

        if self.rank == 0 and self.logger:
            self.logger.info("Training complete.")
        if self.args.ddp:
            DDPHelper.cleanup()


class EncoderTester:
    """
    Tester class to independently run visualizations and evaluate latent features
    without coupling to the training logic.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = config.DEVICE
        self.test_dir = self.args.output_dir
        if self.test_dir is None:
            self.test_dir = config.OUTPUT_DIR / f"test_results_{datetime.datetime.now().strftime('%d_%m_%H_%M')}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model
        self.model = XrayEncoder(
            device=self.device, 
            size=config.IMAGE_SIZE, 
            patch_size=self.args.patch_size
        ).to(self.device)
        
        ckpt_path = self.args.ckpt_dir / "encoder_best.pth"
        if not ckpt_path.exists():
            ckpt_path = self.args.ckpt_dir / "encoder_last.pth"
            
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            # Unwrap DDP metadata if present
            sd = ckpt['model_sd']
            new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
            self.model.load_state_dict(new_sd)
            print(f"Loaded checkpoint for testing from {ckpt_path}")
        else:
            print(f"Warning: No valid checkpoint found in {self.args.ckpt_dir}. Using untrained weights.")
            
        self.model.eval()

        # Build Dataset
        data_dir = self.args.data_dir
        if self.args.on_the_fly:
            self.dataset = MultiPatientDRRDataset(
                data_dir=data_dir,
                device='cpu',
                size=config.IMAGE_SIZE,
                patient_ids=(7, 11)
            )
        else:
            self.dataset = DRRMetadataDataset(root_dir=data_dir)

    def save_visualization(self, patient_id: str, loss: float, original: torch.Tensor, mask: torch.Tensor, reconstructed: torch.Tensor, img_idx: int):
        """
        Saves a side-by-side comparison of original, mask overlay, and reconstructed images.
        
        Args:
            patient_id (str): Identifier for the patient.
            loss (float): Loss value for this sample.
            original (torch.Tensor): Original DRR image tensor.
            mask (torch.Tensor): Patch mask tensor.
            reconstructed (torch.Tensor): Reconstructed image tensor.
        """
        save_dir = self.test_dir / patient_id
        save_dir.mkdir(parents=True, exist_ok=True)

        orig_img = original.detach().cpu().squeeze().numpy()
        mask_img = mask.detach().cpu().squeeze().numpy()
        recon_img = reconstructed.detach().cpu().squeeze().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Encoder Reconstrcution Test | Loss: {loss:.6f}", fontsize=16)
        
        # 1. Original
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # 2. Masked (Original with overlay)
        axes[1].imshow(orig_img, cmap='gray')
        overlay = np.zeros((*orig_img.shape, 4))
        # Masked = Red
        overlay[mask_img == 1] = [1.0, 0.0, 0.0, 0.3]
        # Visible = Blue
        overlay[mask_img == 0] = [0.0, 0.0, 1.0, 0.4]
        axes[1].imshow(overlay)
        axes[1].set_title("Masked")
        axes[1].axis('off')
        
        # 3. Reconstructed
        axes[2].imshow(recon_img, cmap='gray')
        axes[2].set_title("Reconstructed")
        axes[2].axis('off')

        save_file = save_dir / f"reconstruction_{img_idx:02d}.png"
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    def run_all_tests(self, num_per_patient=5):
        """
        Main testing pipeline. Extracts DRRs, saves visualizations, measures feature quality.
        """
        print(f"Starting testing. Collecting {num_per_patient} DRRs per patient.")
        
        # Dictionary to store mapping of patient -> list of latents
        patient_latents = defaultdict(list)
        
        # 1. Grab 5 deterministic DRRs from the dataset per patient
        with torch.no_grad():
            if self.args.on_the_fly:
                # MultiPatientDRRDataset strategy
                for p_idx, entry in enumerate(self.dataset.entries):
                    patient_id = f"patient_{p_idx}"
                    subj, drr, grids, valid_indices = entry
                    
                    seed = 42
                    indices = np.random.RandomState(seed).choice(
                        len(valid_indices), min(num_per_patient, len(valid_indices)), replace=False
                    )
                    
                    for i, idx in enumerate(indices):
                        pose_idx = valid_indices[idx]
                        rot, trans = self.dataset._index_to_pose(grids, pose_idx, self.device)
                        
                        # Generate projection
                        from src.data.dataset import _normalize_projection
                        proj = drr(rot, trans, parameterization="euler_angles", convention="ZXY").squeeze(0)
                        proj = _normalize_projection(proj).unsqueeze(0).to(self.device)  # (1, 1, H, W)
                        
                        # Forward pass
                        recon, pixel_mask, latent = self.model(proj, mask_ratio=self.args.mask_ratio)
                        
                        loss = masked_l1_loss(recon, proj, pixel_mask).item()
                        self.save_visualization(patient_id, loss, proj[0], pixel_mask[0], recon[0], i)
                        patient_latents[patient_id].append(latent.detach().cpu())
            else:
                # DRRMetadataDataset strategy
                patient_counts = defaultdict(int)
                for i in range(len(self.dataset)):
                    sample = self.dataset.samples[i] if hasattr(self.dataset, 'samples') else None
                    if sample is not None:
                        # Infer patient id from directory path assuming `patient_X/`
                        patient_id = Path(sample['path']).parent.name
                    else:
                        patient_id = "unknown_patient"
                        
                    if patient_counts[patient_id] < num_per_patient:
                        img = self.dataset[i].unsqueeze(0).to(self.device) # (1, 1, H, W)
                        recon, pixel_mask, latent = self.model(img, mask_ratio=self.args.mask_ratio)
                        
                        loss = masked_l1_loss(recon, img, pixel_mask).item()
                        self.save_visualization(patient_id, loss, img[0], pixel_mask[0], recon[0], patient_counts[patient_id])
                        patient_latents[patient_id].append(latent.detach().cpu())
                        patient_counts[patient_id] += 1
                        
                    # Stop early if we have 5 of every patient
                    # (In a real scenario, we might iterate completely, but this suffices for large subsets)

        print("\n--- Feature Quality Verification ---")
        self.verify_latent_features(patient_latents)

    def verify_latent_features(self, patient_latents_dict):
        """
        Scalable method to evaluate the properties of the learned latents 
        without explicit pair labels. Useful for understanding registration mapping capability.
        """
        # Collect all latents
        all_patient_ids = list(patient_latents_dict.keys())
        if len(all_patient_ids) == 0:
            print("No latents collected to verify.")
            return

        print(f"Collected latents for {len(all_patient_ids)} patients.")
        
        # 1. Feature Norms / Scaling Check
        # Flattens (B, Seq, D) to (B, Features) depending on latent shape.
        flat_latents = []
        for pid in all_patient_ids:
            for l in patient_latents_dict[pid]:
                # latent = l.mean([2,3])
                latent = F.normalize(l, dim=-1)
                flat_latents.append(latent.flatten())
                
        all_features = torch.stack(flat_latents)  # (N, D)
        feature_mean = all_features.mean(dim=0)
        feature_var = all_features.var(dim=0)
        
        print(f"Overall Feature Dimensionality: {all_features.shape[1]}")
        print(f"Mean Feature L2 Norm: {torch.norm(all_features, dim=1).mean().item():.4f}")
        print(f"Active Features (Var > 0.01): {(feature_var > 0.01).sum().item()} / {all_features.shape[1]}")
        
        # 2. Pairwise Cosine Similarity (Intra-Patient vs Inter-Patient)
        # We expect intra-patient similarity to be higher than inter-patient, 
        # indicating it recognizes patient anatomy differences vs just poses.
        print("\nComputing Cosine Similarities...")
        
        intra_sims = []
        inter_sims = []
        
        for i, pid1 in enumerate(all_patient_ids):
            latents1 = patient_latents_dict[pid1]
            
            # Intra-patient
            for l1_idx in range(len(latents1)):
                for l2_idx in range(l1_idx + 1, len(latents1)):
                    sim = F.cosine_similarity(latents1[l1_idx].flatten().unsqueeze(0), 
                                              latents1[l2_idx].flatten().unsqueeze(0))
                    intra_sims.append(sim.item())
                    
            # Inter-patient
            for pid2 in all_patient_ids[i+1:]:
                latents2 = patient_latents_dict[pid2]
                for l1 in latents1:
                    for l2 in latents2:
                        sim = F.cosine_similarity(l1.flatten().unsqueeze(0), 
                                                  l2.flatten().unsqueeze(0))
                        inter_sims.append(sim.item())
                        
        if intra_sims:
            print(f"Average Intra-Patient Similarity: {np.mean(intra_sims):.4f} +/- {np.std(intra_sims):.4f}")
        if inter_sims:
            print(f"Average Inter-Patient Similarity: {np.mean(inter_sims):.4f} +/- {np.std(inter_sims):.4f}")
            
        print("\nFeature verification complete. You can expand this module with t-SNE or specific regression tests.")


def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    """
    Entry point for the distributed worker processes.
    
    Args:
        rank (int): Process rank.
        world_size (int): Total number of processes.
        args (argparse.Namespace): Command line arguments.
    """
    trainer = EncoderTrainer(rank, world_size, args)
    trainer.train()

def main():
    """
    Main entry point of the script parsing args and launching workers.
    """
    args = parse_args()
    
    # Set visible devices BEFORE any torch.cuda calls or spawning
    set_visible_devices(args.gpus)
    
    # Test mode
    if args.test:
        print("Running in Test Mode...")
        tester = EncoderTester(args)
        tester.run_all_tests(num_per_patient=5)
    # Train mode
    else:
        if args.ddp:
            # Auto-detect distributed availability
            if torch.cuda.is_available():
                world_size = torch.cuda.device_count()
                print(f"Launching DDP training on {world_size} GPUs.")
                DDPHelper.spawn(train_worker, world_size, args=(args,))
            else:
                print("No CUDA device found. Running on CPU (single process).")
                train_worker(0, 1, args)
        else:
            train_worker(0, 1, args)

if __name__ == "__main__":
    main()
