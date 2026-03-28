import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import numpy as np

from diffdrr.drr import DRR
from diffdrr.data import read

from src.core.pose_regressor import PoseRegressor
from src.data.dataset import DRRMetadataDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices
import src.utils.config as config

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
    
    parser.add_argument("--ddp", action="store_true", help="Use DDP parallelization")
    parser.add_argument("--test", action="store_true", help="Disables backwards prop triggering isolated tester loop runs.")

    return parser.parse_args()

# --- Rotational Math Utilities ---

def euler_angles_to_matrix(euler_angles, convention="ZXY"):
    """
    Convert (B, 3) Euler angles in radians to a (B, 3, 3) rotation matrix.
    Convention "ZXY" logic applies sequential product: Rz * Rx * Ry
    """
    z, x, y = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    
    rx = torch.stack([
        torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
        torch.zeros_like(x), cx, -sx,
        torch.zeros_like(x), sx, cx
    ], dim=1).reshape(-1, 3, 3)
    
    ry = torch.stack([
        cy, torch.zeros_like(y), sy,
        torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y),
        -sy, torch.zeros_like(y), cy
    ], dim=1).reshape(-1, 3, 3)
    
    rz = torch.stack([
        cz, -sz, torch.zeros_like(z),
        sz, cz, torch.zeros_like(z),
        torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)
    ], dim=1).reshape(-1, 3, 3)
    
    if convention == "ZXY":
        return torch.bmm(rz, torch.bmm(rx, ry))
    return torch.bmm(rx, torch.bmm(ry, rz)) # Fallback implementation

def matrix_to_rotation_6d(matrix):
    """
    Grab the first two columns spanning the continuous space.
    matrix: (B, 3, 3)
    Returns: (B, 6)
    """
    return matrix[:, :, :2].reshape(-1, 6)

def rotation_6d_to_matrix(d6):
    """
    Differentiable step to form a valid orthogonal 3x3 rotation matrix using Zhou's Continuous 6D formula.
    d6: (B, 6) Raw coordinates of the 2 orthogonal basis vectors representation 
    Returns: (B, 3, 3) Orientation matrix
    """
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]
    
    x = F.normalize(x_raw, dim=1)
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, dim=1)
    y = torch.cross(z, x, dim=1)
    
    return torch.stack((x, y, z), dim=-1)

def compute_geodesic_distance(R1, R2):
    """
    Calculates Geodesic difference in Radians mapped accurately upon SO(3) domain
    R1, R2: Both sizes (B, 3, 3)
    """
    R = torch.bmm(R1, R2.transpose(1, 2))
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta)

def matrix_to_euler_angles(matrix, convention="ZXY", eps=1e-6):
    """
    Convert rotation matrix (B, 3, 3) to Euler angles (B, 3)
    matching euler_angles_to_matrix with ZXY convention:
        R = Rz * Rx * Ry
    
    Returns angles in radians: (z, x, y)
    """
    if convention != "ZXY":
        raise NotImplementedError(f"Convention {convention} not supported")

    R = matrix
    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # x = asin(r21)
    x = torch.asin(torch.clamp(r21, -1.0 + eps, 1.0 - eps))
    cx = torch.cos(x)

    # Detect gimbal lock
    gimbal_lock = torch.abs(cx) < eps

    # Standard case
    z = torch.atan2(-r01, r11)
    y = torch.atan2(-r20, r22)

    # Gimbal lock fallback
    z_gl = torch.atan2(r10, r00)
    y_gl = torch.zeros_like(y)

    z = torch.where(gimbal_lock, z_gl, z)
    y = torch.where(gimbal_lock, y_gl, y)

    return torch.stack([z, x, y], dim=1)

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
        
        self.model = PoseRegressor().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        self.trans_weight = args.trans_weight
        
        # State tracking
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'loss': [], 'rot_loss': [], 'trans_loss': []}

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger("Train Regressor", self.args.ckpt_dir / f"Regressor_{timestamp}.log")
        self.logger.info(f"Starting Pose Regressor Training")
        self.logger.info(f"Arguments: {vars(args)}")

        if args.resume:
            self.load_checkpoint(args.resume)
            
        self.setup_dataloaders()

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

    def setup_dataloaders(self):
        """Prepares dataloader containing purely training DRR images."""
        self.logger.info(f"Preparing Training Dataset from: {self.args.data_dir}")
        train_dataset = DRRMetadataDataset(root_dir=self.args.data_dir, return_pose=True)
        
        self.sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=True
        ) if self.args.ddp else None

        self.loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            sampler=self.sampler, 
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=(self.sampler is None)
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def plot_loss_curve(self, save_path: Path):
        """
        Plots the training loss curves (total, rotational, translational) and saves to file.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['loss']) + 1)
        
        plt.plot(epochs, self.history['loss'], label='Total Loss', marker='o', color='blue')
        plt.plot(epochs, self.history['rot_loss'], label='Rotational Loss', marker='x', linestyle='--', color='orange')
        plt.plot(epochs, self.history['trans_loss'], label='Translational Loss', marker='x', linestyle='--', color='green')
        
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
        total_loss, total_rot_loss, total_trans_loss = 0.0, 0.0, 0.0

        pbar = tqdm(self.train_loader, desc="Minibatch Progression", leave=False)
        for images, poses in pbar:
            images, poses = images.to(self.device), poses.to(self.device)
            
            euler_gt = poses[:, :3]
            trans_gt = poses[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            self.optimizer.zero_grad()
            
            rot_6d_pred, trans_pred = self.model(images)
            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
            
            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            # loss_trans = F.smooth_l1_loss(trans_pred, trans_gt)
            trans_scale = 100.0
            loss_trans = F.smooth_l1_loss(trans_pred / trans_scale, trans_gt / trans_scale)
            
            loss = loss_rot + (self.trans_weight * loss_trans)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_rot_loss += loss_rot.item()
            total_trans_loss += loss_trans.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Rot_Loss": f"{loss_rot.item():.4f}", "Trans_Loss": f"{loss_trans.item():.4f}"})
            
        N = len(self.train_loader)
        return total_loss / N, total_rot_loss / N, total_trans_loss / N

    def run(self):
        """Execution wrapper handling epoch iteration, loss storing, model snapshots and visualizations."""
        self.logger.info("Training sequence started.")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            self.logger.info(f"--- Epoch [ {epoch+1} / {self.args.epochs} ] Started ---")
            
            # Sub-routine handling iteration
            loss, rot_loss, trans_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch+1} Results -> Total Loss: {loss:.4f} | Rot Loss: {rot_loss:.4f} | Trans Loss: {trans_loss:.4f}")
            
            # Append local histories directly
            self.history['loss'].append(loss)
            self.history['rot_loss'].append(rot_loss)
            self.history['trans_loss'].append(trans_loss)
            
            # Save visual outputs showing convergence trend
            self.plot_loss_curve(self.ckpt_dir / "regressor_loss_curve.png")
            
            # Standard checkpoint state
            state = {
                "model": self.model.state_dict(), 
                "epoch": epoch, 
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "history": self.history
            }
            
            # Overwrite last model
            last_pth = self.ckpt_dir / "regressor_last.pth"
            torch.save(state, last_pth)
            self.logger.info(f"Saved latest model tracking to {last_pth}")
            
            # Overwrite best model if improved tracking
            if loss < self.best_loss:
                self.best_loss = loss
                state["best_loss"] = self.best_loss
                best_pth = self.ckpt_dir / "regressor_best.pth"
                torch.save(state, best_pth)
                self.logger.info(f"*** New best model saved with Training Loss {self.best_loss:.4f} ***")

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
        self.logger = setup_logger("Test Regressor", Path(args.ckpt_dir) / f"Regressor_{timestamp}.log")
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
        test_dataset = DRRMetadataDataset(root_dir=test_dir_bound, return_pose=True) 
        self.dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def save_visualization(self, patient_id: str, loss: float, original: torch.Tensor, image_path: str, rot_gt, rot_pred, trans_gt, trans_pred, d6_rot=None):
        """
        Saves a visual overlay mapping the baseline DRR image with its corresponding truth vs generated inference bounds in angles/mm.
        """
        save_dir = self.output_dir / f"patient_{patient_id.item()}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load DRR
        ct_path = f"/mnt/storage/users/amiry/git/Thesis/datasets/new_data/patient_{patient_id:02}/ct.nii.gz"
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

        visualize_idxs = torch.randint(0, len(self.dataloader), size = (5,))

        pbar = tqdm(self.dataloader, desc="Testing Evaluation", leave=False)
        for batch_idx, (images, poses, samples) in enumerate(pbar):
            images, poses = images.to(self.device), poses.to(self.device)
            
            euler_gt = poses[:, :3]
            trans_gt = poses[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")
            
            rot_6d_pred, trans_pred = self.model(images)
            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
            rot_pred_euler = matrix_to_euler_angles(rot_matrix_pred)
            
            rot_dists = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt) # (B)
            trans_dists = torch.norm(trans_pred - trans_gt, dim=1)                # (B)
            
            rot_dists_deg = torch.rad2deg(rot_dists)
            
            # Draw visualizations
            if batch_idx in visualize_idxs:
                for v_idx in range(min(num_vis_per_batch, images.size(0))):
                    # For visualization, calculate Euler logic natively out of pred matrix or track direct geodesic, simplifying to GT tracks overlaying limits.
                    # global_idx = (batch_idx * self.args.batch_size) + v_idx
                    loss_v = rot_dists[v_idx].item() + (self.args.trans_weight * F.smooth_l1_loss(trans_pred[v_idx].unsqueeze(0), trans_gt[v_idx].unsqueeze(0)).item())
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

        all_rot_errs = torch.cat(all_rot_errs)
        all_trans_errs = torch.cat(all_trans_errs)
        
        mean_rot_err = all_rot_errs.mean().item()
        mean_trans_err = all_trans_errs.mean().item()
        
        success_mask = (all_rot_errs < success_rot_deg) & (all_trans_errs < success_trans_mm)
        success_rate = success_mask.float().mean().item() * 100.0
        
        return {
            "mean_rot_err_deg": mean_rot_err,
            "mean_trans_err_mm": mean_trans_err,
            f"success_rate_{success_rot_deg}deg_{success_trans_mm}mm": success_rate
        }

    def run(self):
        """Runner logic initiating stand-alone checks without backwards prop tracking"""
        self.logger.info("Standalone Evaluation Sequence Initialization.")
        metrics = self.evaluate()
        self.logger.info(f"Test Phase Aggregation Metrics: {metrics}")

def main():
    args = parse_args()

    # Pre-configure explicit environment scopes
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Init basic logging streams tracking progressions context natively
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    if args.test:
        # logging.info("--> Operating in TEST-ONLY mode: Training components bypassed.")
        tester = Tester(args)
        tester.run()
    else:
        # logging.info("--> Operating in TRAIN mode: Separated isolated training loop.")
        trainer = Trainer(args)
        trainer.run()

if __name__ == "__main__":
    main()
