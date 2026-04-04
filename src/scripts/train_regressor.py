import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import logging
import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_euler_angles

from src.core.pose_regressor import PoseRegressor
from src.data.dataset import DRRMetadataDataset, RepeatDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger
from src.utils.image_processing import RandomGamma, RandomGaussianNoise, RandomGaussianBlur, RandomContrast, RandomSpatialJitter
# from src.utils.image_processing import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_euler_angles
from src.utils.loss import poseConsistencyLoss, compute_geodesic_distance
import src.utils.config as config

from huggingface_hub import snapshot_download

def parse_args():
    """
    Parses command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser("Train Supervised Pose Regressor")
    parser.add_argument("--data_dir", type=str, default=Path(config.DATA_DIR), help="Directory encompassing patient folders with meta files.")
    parser.add_argument("--output_dir", type=str, default=Path(config.OUTPUT_DIR), help="Root dir logic for outputs.")
    parser.add_argument("--ckpt_dir", type=str, default=Path(config.CKPT_DIR), help="Path retaining epoch weight sets.")
    parser.add_argument("--resume", type=str, default=None, help="Continue sequence from exact path.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Constraint scale")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate target AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Decay AdamW execution")
    parser.add_argument("--epochs", type=int, default=100, help="Cycles")
    parser.add_argument("--trans_weight", type=float, default=0.5, help="Configurable lambda smoothing translation domain.")
    parser.add_argument("--repeats", type=int, default=0, help="Use augmentations, duplicate each input #repeats times")
    
    parser.add_argument("--ddp", action="store_true", help="Use DDP parallelization")
    parser.add_argument("--test", action="store_true", help="Disables backwards prop triggering isolated tester loop runs.")

    return parser.parse_args()

# --- Training & Testing ---

class Trainer:
    """
    Handles supervision loops predicting explicit parameters targeting ground truth translation 
    vector and 6D bounded rotations. Purely isolates training logic over training data.
    """
    def __init__(self, args):
        self.args = args
        self.device = config.DEVICE
        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = PoseRegressor(dropout=0.5).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
       
        # State tracking
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'loss': [], 'rot_loss': [], 'trans_loss': []}

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger("Train Regressor", Path(self.args.ckpt_dir) / f"Regressor_{timestamp}.log")
        self.logger.info(f"Starting Pose Regressor Training")
        self.logger.info(f"Arguments: {vars(args)}")

        if args.resume:
            self.load_checkpoint(args.resume)
            
        self.trans_weight = args.trans_weight * torch.tensor([1.0, 0.2, 1.0], device=self.device)
        # self.trans_scale = 50.0
        self.trans_scale = torch.tensor([40.0, 50.0, 40.0], device=self.device) 
        self.repeats = args.repeats
        self.setup_dataloaders()

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,       # 10% warmup
            anneal_strategy='cos'
        )

        self.transforms = T.Compose([
            RandomGamma(range=(0.5, 2.0)),                 # simulates different dose/kV settings
            RandomGaussianNoise(std_range=(0, 0.05)),      # simulates quantum noise
            RandomGaussianBlur(sigma_range=(0.0001, 1.5)), # simulates MTF differences
            RandomContrast(range=(0.7, 1.3)),              # simulates detector response variation
            RandomSpatialJitter(),
        ])

    def load_checkpoint(self, path: Path):
        """Loads model, optimizer, history and tracks starting epoch."""
        if path.exists():
            self.logger.info(f"Loading previous state bounds: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.start_epoch = ckpt["epoch"] + 1
                if "best_loss" in ckpt:
                    self.best_loss = ckpt["best_loss"]
                if "history" in ckpt:
                    self.history = ckpt["history"]
                self.logger.info(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.5f}")
        else:
            self.logger.warning(f"Checkpoint {path} not found, starting from scratch.")

    def split_dataset(self, train_p=0.7, val_p=0.1, test_p=0.2):
        ids = sorted(int(name.split("_")[1]) for name in os.listdir(self.args.data_dir) if name.startswith("patient_"))
        num_ids = len(ids)
        torch.manual_seed(42)
        perm = torch.randperm(num_ids).tolist()
        ids = [ids[i] for i in perm]  # apply permutation
        
        num_train_ids = int(num_ids * train_p)
        if test_p > 0:
            num_val_ids = int(num_ids * val_p)
            num_test_ids = int(num_ids - num_train_ids - num_val_ids)
            return ids[:num_train_ids], ids[num_train_ids:num_val_ids], ids[num_val_ids:]
        
        return ids[:num_train_ids], ids[num_train_ids:]

    def setup_dataloaders(self):
        """Prepares dataloader containing purely training DRR images."""
        self.logger.info(f"Preparing Training Dataset from: {self.args.data_dir}")

        transforms = None
        # ids = list(range(7,13))
        # ids_train, ids_val = ids[:-1], ids[-1]
        ids_train, ids_val = self.split_dataset(train_p=0.9, val_p=0.1, test_p=0)
        train_dataset = DRRMetadataDataset(root_dir=self.args.data_dir, transform=transforms, return_pose=True, valid_ids=ids_train)
        val_dataset = DRRMetadataDataset(root_dir=self.args.data_dir, transform=transforms, return_pose=True, valid_ids=ids_val)
        
        # train_dataset = RepeatDataset(base_dataset, repeats=self.repeats)
        # train_dataset = base_dataset

        # If consistency loss is used devide batch_size by repeats
        batch_size = (self.args.batch_size // self.repeats) if self.repeats > 1 else self.args.batch_size

        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=True
        ) if self.args.ddp else None
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        ) if self.args.ddp else None

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=(train_sampler is None)
        )

        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def plot_loss_curve(self, save_path: Path):
        """
        Plots the training loss curves (total, rotational, translational) and saves to file.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.plot(epochs, self.history['train_loss'], label='Total Loss', marker='o', color='blue')
        plt.plot(epochs, self.history['train_rot_loss'], label='Rotational Loss', marker='x', linestyle='--', color='orange')
        plt.plot(epochs, self.history['train_trans_loss'], label='Translational Loss', marker='x', linestyle='--', color='green')
        plt.plot(epochs, self.history['train_const_loss'], label='Consistency Loss', marker='x', linestyle='--', color='purple')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss', marker='o', linestyle='--', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Pose Regressor Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def train_epoch(self):
        """Processes a single epoch, isolating model gradients updates and aggregating metrics."""
        self.model.train()
        total_loss, total_rot_loss, total_trans_loss, total_const_loss = 0.0, 0.0, 0.0, 0.0

        pbar = tqdm(self.train_loader, desc="Minibatch Progression", leave=False)
        for images, poses_gt, _ in pbar:
            images, poses_gt = images.to(self.device), poses_gt.to(self.device)
            
            if self.repeats > 1:
                B = images.shape[0]
                # poses_gt = poses_gt.view(B, 1, -1).expand(B, self.repeats, -1).reshape(B * self.repeats, -1)
                poses_gt = poses_gt.repeat_interleave(self.repeats, dim=0)
                images_augmented = images.repeat_interleave(self.repeats, dim=0)
                images_repeated = self.transforms(images_augmented)
                # images_augmented = images.repeat_interleave(self.repeats - 1, dim=0)

                # images_augmented = self.transforms(images_augmented)
                
                # images_augmented = images_augmented.view(B, self.repeats - 1, *images.shape[1:]) #(B, 3, C, H, W)
                # images_all = torch.cat([images.unsqueeze(1), images_augmented], dim=1) #(B, 4, C, H, W)
                # images_repeated = images_all.view(B * self.repeats, *images.shape[1:]) # (4B, C, H, W)
            else:
                images_repeated = images

            euler_gt = poses_gt[:, :3]
            trans_gt = poses_gt[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            self.optimizer.zero_grad()
            
            rot_6d_pred, trans_pred = self.model(images_repeated)
            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
            
            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            # loss_trans = F.smooth_l1_loss(trans_pred, trans_gt)
            
            # loss_trans = F.smooth_l1_loss(trans_pred / self.trans_scale, trans_gt / self.trans_scale)
            loss_trans_by_axis = F.smooth_l1_loss(trans_pred, trans_gt, reduction='none')  / self.trans_scale
            loss_trans = loss_trans_by_axis.mean()

            if self.repeats > 1:
                consistency_loss = poseConsistencyLoss(torch.cat([rot_6d_pred, trans_pred], dim=1))
            else:
                consistency_loss = torch.tensor(0.0, device=self.device)

            loss = loss_rot + (self.trans_weight * loss_trans_by_axis).mean() + (0.3 * consistency_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_rot_loss += loss_rot.item()
            total_trans_loss += loss_trans.item()
            total_const_loss += consistency_loss.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Rot_Loss": f"{loss_rot.item():.4f}", "Trans_Loss": f"{loss_trans.item():.4f}", "Const_Loss": f"{consistency_loss.item():.4f}"})
            
        N = len(self.train_loader)
        return total_loss / N, total_rot_loss / N, total_trans_loss / N, total_const_loss / N

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss, total_rot_loss, total_trans_loss = 0.0, 0.0, 0.0

        for images, poses_gt, _ in self.val_loader:
            images = images.to(self.device)
            poses_gt = poses_gt.to(self.device)

            euler_gt = poses_gt[:, :3]
            trans_gt = poses_gt[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            rot_6d_pred, trans_pred = self.model(images)
            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            # loss_trans = F.smooth_l1_loss(trans_pred / self.trans_scale, trans_gt / self.trans_scale)
            loss_trans_by_axis = F.smooth_l1_loss(trans_pred, trans_gt, reduction='none')  / self.trans_scale
            loss_trans = loss_trans_by_axis.mean()

            loss = loss_rot + (self.trans_weight * loss_trans_by_axis).mean()

            total_loss += loss.item()
            total_rot_loss += loss_rot.item()
            total_trans_loss += loss_trans.item()

        N = len(self.val_loader)
        return total_loss / N, total_rot_loss / N, total_trans_loss / N

    def run(self):
        """Execution wrapper handling epoch iteration, loss storing, model snapshots and visualizations."""
        self.logger.info("Training sequence started.")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self.logger.info(f"--- Epoch [ {epoch+1} / {self.args.epochs} ] Started ---")
            
            # Sub-routine handling iteration
            train_loss, train_rot_loss, train_trans_loss, train_const_loss = self.train_epoch()
            self.logger.info(f"[Train] Results -> Total Loss: {train_loss:.4f} | Rot Loss: {train_rot_loss:.4f} | Trans Loss: {train_trans_loss:.4f} | Const Loss: {train_const_loss:.4f}")
            
            # VALIDATE
            val_loss, val_rot_loss, val_trans_loss = self.validate()

            self.logger.info(f"[Val] Results -> Total Loss: {val_loss:.4f} | Rot Loss: {val_rot_loss:.4f} | Trans Loss: {val_trans_loss:.4f}")

            # --------------------
            # STORE HISTORY
            # --------------------
            self.history.setdefault('train_loss', []).append(train_loss)
            self.history.setdefault('train_rot_loss', []).append(train_rot_loss)
            self.history.setdefault('train_trans_loss', []).append(train_trans_loss)
            self.history.setdefault('train_const_loss', []).append(train_const_loss)

            self.history.setdefault('val_loss', []).append(val_loss)
            self.history.setdefault('val_rot_loss', []).append(val_rot_loss)
            self.history.setdefault('val_trans_loss', []).append(val_trans_loss)

            # --------------------
            # PLOT
            # --------------------
            self.plot_loss_curve(self.ckpt_dir / "regressor_loss_curve.png")

            # --------------------
            # SAVE CHECKPOINT
            # --------------------
            state = {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "history": self.history
            }

            # Save latest
            last_pth = self.ckpt_dir / "regressor_last.pth"
            torch.save(state, last_pth)
            self.logger.info(f"Saved latest model to {last_pth}")

            # --------------------
            # BEST MODEL (based on validation!)
            # --------------------
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                state["best_loss"] = self.best_loss

                best_pth = self.ckpt_dir / "regressor_best.pth"
                torch.save(state, best_pth)

                self.logger.info(f"*** New best model saved with VAL Loss {self.best_loss:.4f} ***")


        self.logger.info("Training sequence completed.")

class Tester:
    """
    Independent tester validation logic isolating test datasets from the training bounds.
    """
    def __init__(self, args):
        self.args = args
        self.device = config.DEVICE
        self.output_dir = Path(getattr(config, "OUTPUT_DIR", args.output_dir)) / f"test_results_{datetime.datetime.now().strftime('%d_%m_%H_%M')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger("Test Regressor", Path(self.output_dir) / f"Regressor_{timestamp}.log")
        self.logger.info(f"Starting Pose Regressor Testing")
        self.logger.info(f"Arguments: {vars(args)}")

        self.model = PoseRegressor().to(self.device)
        
        # Prefer best model but fallback to last
        ckpt_path = Path(args.ckpt_dir) / "regressor_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(args.ckpt_dir) / "regressor_last.pth"
            
        if args.resume: # Override specific explicitly given weights priority
            ckpt_path = Path(args.resume)

        if ckpt_path.exists():
            self.logger.info(f"Loading checkpoint for testing from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
        else:
            self.logger.warning(f"No valid checkpoint found. Continuing with randomly initialized weights!")

        self.setup_dataloader()

    def setup_dataloader(self):
        """Isolates testing logic bounds specifically reading unseen bounds if configured"""
        test_dir_bound = self.args.data_dir
        self.logger.info(f"Preparing Testing Dataset from: {test_dir_bound}")
        ids = list(range(13,15))
        test_dataset = DRRMetadataDataset(root_dir=test_dir_bound, return_pose=True, valid_ids=ids) 
        self.dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def save_visualization(self, patient_id: str, loss: float, original: torch.Tensor, image_path: str, rot_gt, rot_pred, trans_gt, trans_pred, d6_rot=None):
        """
        Saves a visual overlay mapping the baseline DRR image with its corresponding truth vs generated inference bounds in angles/mm.
        """
        save_dir = self.output_dir / f"patient_{patient_id.item()}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load DRR
        ct_path = Path(self.args.data_dir) / f"patient_{patient_id:02d}/ct.nii.gz"
        subject = read(volume=str(ct_path), orientation="AP", center_volume=True)
        drr = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX)
        projection = drr(rot_pred.unsqueeze(0).to('cpu'), trans_pred.unsqueeze(0).to('cpu'), parameterization="euler_angles", convention="ZXY")
        # print(d6_rot)
        # projection = drr(d6_rot, trans_pred, parameterization="rotation_6d")

        img = original.detach().cpu().squeeze().numpy()
        projection = 1 - projection.detach().cpu().squeeze().numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title(f"GT Pose Projection")
        
        # Map values cleanly
        pred_angles = np.rad2deg(rot_pred.cpu().numpy())
        gt_angles = np.rad2deg(rot_gt)
        
        info_text = (
            f"Loss: {loss:.4f}\n"
            f"GT Rot (deg): [{gt_angles[0]:.1f}, {gt_angles[1]:.1f}, {gt_angles[2]:.1f}] | Trans (mm): [{trans_gt[0]:.1f}, {trans_gt[1]:.1f}, {trans_gt[2]:.1f}]\n"
            f"PR Rot (deg): [{pred_angles[0]:.1f}, {pred_angles[1]:.1f}, {pred_angles[2]:.1f}] | Trans (mm): [{trans_pred[0]:.1f}, {trans_pred[1]:.1f}, {trans_pred[2]:.1f}]"
        )
        # ax[0].set_title(info_text, fontsize=9, loc='left')

        ax[1].imshow(projection, cmap='gray')
        ax[1].axis('off')
        ax[1].set_title(f"Predicted Pose Projection")

        # ✅ Main title (above everything)
        fig.suptitle(info_text, fontsize=10)

        serial = image_path.split("drr_")[-1].split(".png")[0]
        serial = int(serial)

        save_file = save_dir / f"pose_eval_{serial:03d}.png"
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def evaluate(self, success_rot_deg=5.0, success_trans_mm=10.0, num_vis_per_batch=1):
        """
        Runs comprehensive inference validation outputting constraint success scaling percentages
        and generating selective inference visuals map overlays representing regressor accuracy contexts.
        """
        self.model.eval()
        
        all_rot_errs = []
        all_trans_errs = []
        xz_trans_errs = []

        visualize_idxs = torch.randint(0, len(self.dataloader), size = (5,))

        pbar = tqdm(self.dataloader, desc="Testing Evaluation", leave=False)
        for batch_idx, (images, poses_gt, samples) in enumerate(pbar):
            images, poses_gt = images.to(self.device), poses_gt.to(self.device)
            
            euler_gt = poses_gt[:, :3]
            trans_gt = poses_gt[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")
            
            rot_6d_pred, trans_pred = self.model(images)
            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
            rot_pred_euler = matrix_to_euler_angles(rot_matrix_pred, convention="ZXY")
            
            rot_dists = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt)           # (B)
            trans_dists = torch.norm(trans_pred - trans_gt, dim=1)                          # (B)
            trans_dists_xz = torch.norm(trans_pred[:, [0,2]] - trans_gt[:, [0,2]], dim=1)  # (B)
            
            rot_dists_deg = torch.rad2deg(rot_dists)
            
            # Draw visualizations
            if batch_idx in visualize_idxs:
                for v_idx in range(min(num_vis_per_batch, images.size(0))):
                    # For visualization, calculate Euler logic natively out of pred matrix or track direct geodesic, simplifying to GT tracks overlaying limits.
                    loss_v = rot_dists[v_idx].item() + (self.args.trans_weight * F.smooth_l1_loss(trans_pred[v_idx].unsqueeze(0) / 50.0, trans_gt[v_idx].unsqueeze(0)).item() / 50.0)
                    self.save_visualization(
                        patient_id=samples['id'][v_idx], 
                        loss=loss_v, 
                        original=images[v_idx], 
                        image_path=samples['path'][v_idx],
                        rot_gt=euler_gt[v_idx].cpu().numpy(), 
                        rot_pred=rot_pred_euler[v_idx],
                        trans_gt=trans_gt[v_idx].cpu().numpy(), 
                        trans_pred=trans_pred[v_idx],
                        d6_rot = rot_6d_pred[v_idx].cpu().numpy(),
                    )

            all_rot_errs.append(rot_dists_deg.cpu())
            all_trans_errs.append(trans_dists.cpu())
            xz_trans_errs.append(trans_dists_xz.cpu())

        all_rot_errs = torch.cat(all_rot_errs)
        all_trans_errs = torch.cat(all_trans_errs)
        xz_trans_errs = torch.cat(xz_trans_errs)
        
        mean_rot_err = all_rot_errs.mean().item()
        mean_trans_err = all_trans_errs.mean().item()
        mean_xz_trans_err = xz_trans_errs.mean().item()
        
        success_mask = (all_rot_errs < success_rot_deg) & (all_trans_errs < success_trans_mm)
        success_rate = success_mask.float().mean().item() * 100.0
        
        xz_success_mask = (all_rot_errs < success_rot_deg) & (xz_trans_errs < success_trans_mm)
        xz_success_rate = xz_success_mask.float().mean().item() * 100.0

        # Group errors by patient_id
        per_patient = {}
        for err_r, err_t, err_xz, pid in zip(all_rot_errs, all_trans_errs, xz_trans_errs, [13, 14]):
            per_patient.setdefault(pid, []).append((err_r, err_t, err_xz))
        for pid, errs in per_patient.items():
            r_errs = torch.tensor([e[0] for e in errs])
            t_errs = torch.tensor([e[1] for e in errs])
            xz_errs = torch.tensor([e[2] for e in errs])
            print(f"Patient {pid}: rot={r_errs.mean():.1f}° trans={t_errs.mean():.1f}mm xz={xz_errs.mean():.1f}mm")

        return {
            "mean_rot_err_deg": mean_rot_err,
            "mean_trans_err_mm": mean_trans_err,
            f"success_rate_{success_rot_deg}deg_{success_trans_mm}mm": success_rate,
            f"xz_success_rate_{success_rot_deg}deg_{success_trans_mm}mm": xz_success_rate
        }

    def run(self, success_rot_deg=5.0, success_trans_mm=10.0, num_vis_per_batch=1):
        """Runner logic initiating stand-alone checks without backwards prop tracking"""
        self.logger.info("Standalone Evaluation Sequence Initialization.")
        metrics = self.evaluate(success_rot_deg, success_trans_mm, num_vis_per_batch)
        self.logger.info(f"Test Phase Aggregation Metrics: {metrics}")

def main():
    args = parse_args()

    # Pre-configure explicit environment scopes
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
        
    if args.test:
        tester = Tester(args)
        tester.run(success_rot_deg=10.0, success_trans_mm=20.0)
    else:
        trainer = Trainer(args)
        trainer.run()

if __name__ == "__main__":
    main()
