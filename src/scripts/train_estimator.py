import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import sys
import torch
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import time
import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt

from src.utils import config
from src.core.encoder import XrayEncoder
from src.core.estimator import PositionEstimator
from src.data.dataset import MultiPatientDRRDataset, DRRMetadataDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices
from src.utils.loss import PositionLoss


def parse_args():
    """
    Parses command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train PositionEstimator (Pose Regression)")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    
    # Dataset / Patient
    parser.add_argument("--patient_id", type=int, required=True, help="Patient ID for single-CT initialization inside estimator")
    parser.add_argument("--data_dir", type=Path, default=Path(config.DATA_DIR), help="Root directory containing CT/ and CRM/")
    parser.add_argument("--on_the_fly", action="store_true", help="Generate DRR using MultiPatientDRRDataset while training or use pre-generated DRRMetadataDataset")
    
    # Pretrained Encoder
    parser.add_argument("--ckpt_dir", type=Path, default=Path(config.CKPT_DIR), help="Path to trained encoder checkpoint (required)")
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE, help="Patch size used in encoder")

    # Paths
    parser.add_argument("--output_dir", type=Path, default=Path(config.OUTPUT_DIR), help="Directory to save checkpoints/logs")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")

    # Distributed
    parser.add_argument("--ddp", action="store_false", help="Use DDP parallelization")
    
    return parser.parse_args()


class EstimatorTrainer:
    """
    Trainer class encapsulating the training logic for the Position Estimator.
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logger = setup_logger("train_estimator", self.args.ckpt_dir / f"estimator_{timestamp}.log")
            self.logger.info(f"Starting Estimator Training for Patient {self.args.patient_id} on {world_size} GPUs")
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
            if self.rank == 0 and self.logger:
                self.logger.info("Using MultiPatientDRRDataset (on-the-fly)")
            # MultiPatientDRRDataset provides projections and their corresponding poses
            self.dataset = MultiPatientDRRDataset(
                data_dir=data_dir,
                device='cpu',
                size=config.IMAGE_SIZE,
                patient_ids=(7, 11),
                return_pose=True
            )
        else:
            if self.rank == 0 and self.logger:
                self.logger.info("Using DRRMetadataDataset (pre-computed)")
            # DRRMetadataDataset provides deterministic images and their poses
            self.dataset = DRRMetadataDataset(
                root_dir=data_dir,
                return_pose=True
            )

        # Distribute samples across processes
        self.sampler = DistributedSampler(
            self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        ) if self.args.ddp else None
        
        # Note: if self.sampler is not None, shuffle must be False.
        self.loader = DataLoader(
            self.dataset, 
            batch_size=self.args.batch_size, 
            sampler=self.sampler, 
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=(self.sampler is None)
        )

    def _load_encoder(self, encoder, ckpt_path):
        """
        Safely loads encoder weights handling DDP prefixes.
        
        Args:
            encoder (nn.Module): The encoder model.
            ckpt_path (Path): Path to the encoder checkpoint.
            
        Returns:
            nn.Module: Mode with loaded weights.
        """
        if not ckpt_path or not ckpt_path.exists():
            if self.rank == 0 and self.logger:
                self.logger.warning(f"Encoder checkpoint {ckpt_path} not found! Using random init (Not Recommended).")
            return encoder

        if self.rank == 0 and self.logger:
            self.logger.info(f"Loading encoder from {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=encoder.device, weights_only=False)
        
        sd = ckpt['model_sd'] if 'model_sd' in ckpt else ckpt
        
        new_sd = OrderedDict()
        for k, v in sd.items():
            name = k.replace('module.', '')
            new_sd[name] = v
            
        encoder.load_state_dict(new_sd, strict=True)
        return encoder

    def _setup_model(self):
        """Initializes the estimator model, wraps it in DDP, and sets up the optimizer."""
        # A. Encoder
        encoder = XrayEncoder(
            device=self.device, 
            size=config.IMAGE_SIZE, 
            patch_size=self.args.patch_size
        ).to(self.device)
        
        encoder = self._load_encoder(encoder, self.args.ckpt_dir / "encoder_best.pth")

        # B. Estimator
        ct_file = self.args.data_dir / f"patient_{self.args.patient_id:02d}/ct.nii.gz"
        crm_file = self.args.data_dir / f"patient_{self.args.patient_id:02d}/carm_01.png"
        ct_file = Path(f"/mnt/storage/users/amiry/git/Thesis/datasets/new_data/patient_{self.args.patient_id:02d}/ct.nii.gz")
        crm_file = Path(f"/mnt/storage/users/amiry/git/Thesis/datasets/new_data/patient_{self.args.patient_id:02d}/carm_01.png")
        # ct_file = self.args.data_dir / "CT" / f"{self.args.patient_id:02}.nii.gz"
        # crm_file = self.args.data_dir / "CRM" / f"{self.args.patient_id}.png"
        
        if not ct_file.exists() and self.rank == 0 and self.logger:
            self.logger.error(f"CT file missing: {ct_file}")
            
        # The PositionEstimator needs a template CT for its projection mapping
        self.model = PositionEstimator(
            encoder=encoder,
            dicom_file=str(ct_file),
            crm_path=str(crm_file),
        ).to(self.device)
        
        if self.args.ddp:
            # We use find_unused_parameters=True if parts of the encoder (like decoder) aren't used for regression
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )
        self.criterion = PositionLoss().to(self.device)

    def _resume_checkpoint(self):
        """Loads model and optimizer states if a resume checkpoint path is provided."""
        if self.args.resume:
            ckpt = self.ckpt_manager.load(self.args.resume, self.device)
            if ckpt:
                # Need to gracefully handle DDP or non-DDP wrapped state dicts
                sd = ckpt['model_sd']
                if not self.args.ddp:
                    sd = {k.replace('module.', ''): v for k, v in sd.items()}
                self.model.load_state_dict(sd)
                self.optimizer.load_state_dict(ckpt['optim_sd'])
                self.start_epoch = ckpt['epoch'] + 1
                self.best_loss = ckpt.get('best_loss', float('inf'))
                self.history = ckpt.get('history', {'loss': []})
                if self.rank == 0 and self.logger:
                    self.logger.info(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.5f}")
            else:
                if self.rank == 0 and self.logger:
                    self.logger.warning(f"Checkpoint {self.args.resume} not found.")

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
        plt.title('Estimator Training Loss Curve')
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
            float: Average training loss.
        """
        self.model.train()
        meters = AverageMeter()
        
        if self.rank == 0:
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
        else:
            pbar = self.loader

        for i, (proj_gt, pose_gt) in enumerate(pbar):
            proj_gt = proj_gt.to(self.device, non_blocking=True)
            pose_gt = pose_gt.to(self.device, non_blocking=True)
            
            # proj_gt should be (B, 1, H, W)
            if proj_gt.dim() == 3:
                proj_gt = proj_gt.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Predict Pose from Ground Truth Projection directly
            proj_pred, pose_pred = self.model(proj_gt)
            
            # Compute pose distance error
            loss = self.criterion(pose_pred, pose_gt)
            
            loss.backward()
            self.optimizer.step()
            
            meters.update(loss.item(), proj_gt.size(0))
            
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
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            
            avg_loss = self.train_one_epoch(epoch)
            
            if self.rank == 0:
                self.history['loss'].append(avg_loss)
                self.plot_loss_curve(self.args.ckpt_dir / "estimator_loss.png")
                
                elapsed = time.time() - start_time
                if self.logger:
                    self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} | Loss: {avg_loss:.6f} | Elapsed: {elapsed:.1f}s")
                
                # Checkpointing
                state = {
                    'epoch': epoch,
                    'model_sd': self.model.state_dict(),
                    'optim_sd': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'patient_id': self.args.patient_id,
                    'history': self.history,
                    'config': vars(self.args)
                }
                
                self.ckpt_manager.save(f"estimator{self.args.patient_id:02d}_last", state)
                
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    state['best_loss'] = self.best_loss
                    self.ckpt_manager.save(f"estimator{self.args.patient_id:02d}_best", state)
                    if self.logger:
                        self.logger.info(f"New best model saved with loss {self.best_loss:.6f}")

        if self.rank == 0 and self.logger:
            self.logger.info("Training complete.")
        if self.args.ddp:
            DDPHelper.cleanup()


def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    """
    Entry point for the distributed worker processes.
    
    Args:
        rank (int): Process rank.
        world_size (int): Total number of processes.
        args (argparse.Namespace): Command line arguments.
    """
    trainer = EstimatorTrainer(rank, world_size, args)
    trainer.train()


def main():
    """
    Main entry point of the script parsing args and launching workers.
    """
    args = parse_args()
    
    # Set visible devices BEFORE any torch.cuda calls or spawning
    set_visible_devices(args.gpus)
    
    if args.ddp:
        # Auto-detect distributed availability
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            print(f"Launching DDP Estimator Training on {world_size} GPUs.")
            DDPHelper.spawn(train_worker, world_size, args=(args,))
        else:
            print("No CUDA device found. Running on CPU (single process).")
            train_worker(0, 1, args)
    else:
        print("Running Estimator Training on single CPU/GPU process.")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
