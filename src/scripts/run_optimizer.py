import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import datetime
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_euler_angles

from src.utils import config
from src.utils.training import setup_logger
from src.utils.loss import compute_geodesic_distance
from src.core.registration import PoseRegressorOptimizer, Optimizer
from src.core.pose_regressor import PoseRegressor

def parse_args():
    """
    Parses command line arguments for the optimizer script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run Pose Regressor Optimizer Pipeline")
    
    parser.add_argument("--index", type=int, default=13, help="Patient index ID to load the CT volume")
    parser.add_argument("--data_dir", type=str, default=Path(config.DATA_DIR), help="Explicit path to CT volume (overrides index)")
    parser.add_argument("--output_dir", type=str, default=None, help="Root directory for outputs")
    parser.add_argument("--ckpt_dir", type=str, default=Path(config.CKPT_DIR), help="Path to PoseRegressor weights")
    
    # Optimizer settings
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension for PoseGenerator")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for PoseGenerator")
    parser.add_argument("--iters", type=int, default=250, help="Max optimization iterations per sample")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping during optimization")
    
    # Execution settings
    parser.add_argument("--num_samples", type=int, default=50, help="Number of random poses/images to evaluate")
    parser.add_argument("--num_visualize", type=int, default=5, help="Number of samples to visualize and save")
    
    return parser.parse_args()


class OptimizerPipeline:
    """
    Pipeline for testing the PoseRegressor paired with the Latent Pose Optimizer.
    Processes dynamically generated DRR images, computes an initial pose via PoseRegressor,
    and refines the pose using the Latent Pose Optimizer.
    """
    def __init__(self, args):
        self.args = args
        self.device = config.DEVICE
        
        # Set up output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.OUTPUT_DIR) / f"optimizer_results_{timestamp}" if args.output_dir is None else Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger("Optimizer Pipeline", self.output_dir / "optimization.log")
        self.logger.info("Initializing Optimizer Pipeline...")
        self.logger.info(f"Arguments: {vars(args)}")
        
        # Initialize DRR renderer
        self.logger.info(f"Loading CT volume for patient index {args.index}...")
        self.ct_path = Path(args.data_dir) / f"patient_{args.index:02d}/ct.nii.gz"
        subject = read(volume=str(self.ct_path), orientation="AP", center_volume=True)
        drr = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX).to(self.device)
        
        # Initialize Optimizer (which holds PoseRegressor internally)
        self.logger.info("Initializing Pose Regressor Optimizer...")
        self.optimizer = PoseRegressorOptimizer(
            drr=drr, 
            latent_dim=args.latent_dim, 
            hidden_dim=args.hidden_dim
        ).to(self.device)
        
        # Load weights for PoseRegressor
        self.pid = int(self.args.index)
        self.ckpt_dir = self.args.ckpt_dir
        ckpt_path = Path(self.ckpt_dir) / f"regressor_patient{self.pid:02d}.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(self.ckpt_dir) / "regressor_best.pth"
            if not ckpt_path.exists():
                ckpt_path = Path(self.ckpt_dir) / "regressor_last.pth"
        if ckpt_path.exists():
            self.logger.info(f"Loading PoseRegressor checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model_sd" in ckpt:
                self.optimizer.pose_regressor.load_state_dict(ckpt["model_sd"])
            elif "model" in ckpt:
                self.optimizer.pose_regressor.load_state_dict(ckpt["model"])
            else:
                self.optimizer.pose_regressor.load_state_dict(ckpt)
        else:
            self.logger.warning("No valid PoseRegressor checkpoint provided. Using untrained weights!")

    def generate_random_pose(self):
        """
        Generates a valid random GT pose to be used for rendering.
        
        Returns:
            torch.Tensor: (1, 6) tensor containing Euler angles (Z,X,Y) and translations.
        """
        # Euler angles (Z,X,Y) in radians
        # angle_stds = torch.tensor([0.3, 0.3, 0.3], device=self.device)
        angle_stds = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        euler_angles = torch.randn(3, device=self.device) * angle_stds
        
        # Translations in mm
        # trans_stds = torch.tensor([40.0, 50.0, 40.0], device=self.device)
        trans_stds = torch.tensor([10.0, 10.0, 20.0], device=self.device)
        translation = torch.randn(3, device=self.device) * trans_stds
        translation[1] += 650.0  # Base y-offset typically used in the dataset
        
        return torch.cat([euler_angles, translation], dim=-1).unsqueeze(0)

    def compute_metrics(self, gt_pose, pred_pose):
        """
        Computes geodesic rotation error and L2 translation error between ground truth and prediction.
        
        Args:
            gt_pose (torch.Tensor): Ground truth pose (1, 6)
            pred_pose (torch.Tensor): Predicted pose (1, 6)
            
        Returns:
            tuple: (rot_err_deg, trans_err_mm)
        """
        if gt_pose.shape[1] == 6:
            gt_rot = gt_pose[:, :3]
            gt_trans = gt_pose[:, 3:]
            gt_matrix = euler_angles_to_matrix(gt_rot, convention="ZXY")
        elif gt_pose.shape[1] == 9:
            gt_rot = gt_pose[:, :6]
            gt_trans = gt_pose[:, 6:]
            gt_matrix = rotation_6d_to_matrix(gt_rot)

        if pred_pose.shape[1] == 6:
            pred_rot = pred_pose[:, :3]
            pred_trans = pred_pose[:, 3:]
            pred_matrix = euler_angles_to_matrix(pred_rot, convention="ZXY")
        elif pred_pose.shape[1] == 9:
            pred_rot = pred_pose[:, :6]
            pred_trans = pred_pose[:, 6:]
            pred_matrix = rotation_6d_to_matrix(pred_rot)
        
        rot_dist = compute_geodesic_distance(pred_matrix, gt_matrix)
        rot_err_deg = torch.rad2deg(rot_dist).item()
        
        trans_err_mm = torch.norm(pred_trans - gt_trans, dim=1).item()
        
        return rot_err_deg, trans_err_mm

    def compute_loss(self, gt_image, gt_pose, rot_6d_pred, trans_pred):
            euler_gt = gt_pose[:, :3]
            trans_gt = gt_pose[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            
            trans_scale = torch.tensor([40, 50, 40], device=config.DEVICE)
            loss_trans_by_axis = nn.functional.smooth_l1_loss(trans_pred, trans_gt, reduction='none')  / trans_scale
            loss_trans = loss_trans_by_axis.mean()

            trans_weight = trans_scale = torch.tensor(0.5 * [1.0, 0.2, 1.0], device=config.DEVICE)
            loss = loss_rot + (trans_weight * loss_trans_by_axis).mean()

            return loss, loss_rot, loss_trans

    def save_visualization(self, idx, gt_carm, init_proj, opt_proj, init_gain, final_gain):
        """
        Saves a visualization showing the ground truth C-arm image alongside the 
        initial prediction projection and the optimized projection.
        """
        cols = 3
        fig, axes = plt.subplots(1, cols, figsize=(15, 5))
        
        gt_img = gt_carm[0].squeeze().cpu().numpy()
        init_img = init_proj[0].squeeze().cpu().numpy()
        opt_img = opt_proj[0].squeeze().cpu().numpy()
        
        images = [gt_img, init_img, opt_img]
        titles = [
            "Ground Truth C-arm", 
            f"Init Proj (Gain: {init_gain:.3f})", 
            f"Opt Proj (Gain: {final_gain:.3f})"
        ]
        
        for j, (im, t) in enumerate(zip(images, titles)):
            axes[j].imshow(im, cmap="gray")
            axes[j].set_title(t)
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"sample_{idx:03d}.png", bbox_inches='tight')
        plt.close()

    def find_good_pose(self, max_loss = 10.0):
        for i in range(100):
            gt_pose = self.generate_random_pose()
            projection = self.optimizer.render_drr(gt_pose)
            rot_6d_pred, trans_pred = self.optimizer.pose_regressor(projection)
                    
            euler_gt = gt_pose[:, :3]
            trans_gt = gt_pose[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            trans_scale = 50.0
            trans_weight = 0.5
            loss_trans = nn.functional.smooth_l1_loss(trans_pred / trans_scale, trans_gt / trans_scale)

            loss = loss_rot + trans_weight * loss_trans

            if loss < max_loss:
                break
        print(f"GT pose loss was {loss}")
        return gt_pose

    def run(self):
        """
        Main execution loop. Generates runtime renders, optimizes poses, and aggregates metrics.
        """
        num_samples = self.args.num_samples
        self.logger.info(f"Starting execution for {num_samples} dynamic samples.")
        
        metrics = {
            "init_rot_err": [], "init_trans_err": [],
            "opt_rot_err": [], "opt_trans_err": [],
            "init_gain": [], "opt_gain": []
        }
        
        for i in range(num_samples):
            self.logger.info(f"--- Sample [{i+1}/{num_samples}] ---")
            
            # 1. Generate Ground Truth
            # gt_pose = self.generate_random_pose()
            gt_pose = self.find_good_pose(0.1)
            with torch.no_grad():
                carm = self.optimizer.render_drr(gt_pose)
            self.optimizer.update_carm(carm)
            
            # 2. Optimize
            best_pose, best_proj, steps, init_results, final_results = self.optimizer(
                gt_pose=gt_pose,
                iters=self.args.iters, 
                patience=self.args.patience, 
                verbose=True
            )
            
            # 3. Extract Initial and Final Poses
            init_gain, init_pose, _, _ = init_results
            final_gain, final_pose, _, _ = final_results
            
            # 4. Compute Metrics
            init_rot_err, init_trans_err = self.compute_metrics(gt_pose, init_pose)
            opt_rot_err, opt_trans_err   = self.compute_metrics(gt_pose, final_pose)
            
            # 5. Log Results
            self.logger.info(
                f"Initial -> Gain: {init_gain:.4f} | Rot Err: {init_rot_err:.2f} deg | Trans Err: {init_trans_err:.2f} mm"
            )
            self.logger.info(
                f"Final   -> Gain: {final_gain:.4f} | Rot Err: {opt_rot_err:.2f} deg | Trans Err: {opt_trans_err:.2f} mm"
            )
            self.logger.info(f"Converged in {steps} steps.")
            self.logger.info("-" * 40)
            
            # 6. Store Metrics
            metrics["init_rot_err"].append(init_rot_err)
            metrics["init_trans_err"].append(init_trans_err)
            metrics["opt_rot_err"].append(opt_rot_err)
            metrics["opt_trans_err"].append(opt_trans_err)
            metrics["init_gain"].append(init_gain)
            metrics["opt_gain"].append(final_gain)
            
            # 7. Visualization
            if i < self.args.num_visualize:
                with torch.no_grad():
                    init_proj = self.optimizer.render_drr(init_pose, parameterization="rotation_6d")
                self.save_visualization(
                    idx=i+1, 
                    gt_carm=carm, 
                    init_proj=init_proj, 
                    opt_proj=best_proj,
                    init_gain=init_gain, 
                    final_gain=final_gain
                )
                
        # Aggregate statistics and summarize
        self.logger.info("=== Aggregated Metrics ===")
        self.logger.info(f"Mean Initial Rot Err: {np.mean(metrics['init_rot_err']):.2f} +/- {np.std(metrics['init_rot_err']):.2f} deg")
        self.logger.info(f"Mean Optimized Rot Err: {np.mean(metrics['opt_rot_err']):.2f} +/- {np.std(metrics['opt_rot_err']):.2f} deg")
        self.logger.info(f"Mean Initial Trans Err: {np.mean(metrics['init_trans_err']):.2f} +/- {np.std(metrics['init_trans_err']):.2f} mm")
        self.logger.info(f"Mean Optimized Trans Err: {np.mean(metrics['opt_trans_err']):.2f} +/- {np.std(metrics['opt_trans_err']):.2f} mm")
        self.logger.info(f"Mean Initial Gain: {np.mean(metrics['init_gain']):.4f}")
        self.logger.info(f"Mean Optimized Gain: {np.mean(metrics['opt_gain']):.4f}")
        self.logger.info("Execution complete.")


if __name__ == "__main__":
    # Ensure reproducible pseudo-random behavior optionally
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    args = parse_args()
    pipeline = OptimizerPipeline(args)
    pipeline.run()



class RunFullPipeline:
    """
    Pipeline for testing the PoseRegressor paired with the Latent Pose Optimizer.
    Processes dynamically generated DRR images, computes an initial pose via PoseRegressor,
    and refines the pose using the Latent Pose Optimizer.
    """
    def __init__(self, args):
        self.args = args
        self.device = config.DEVICE
        
        # Set up output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.OUTPUT_DIR) / f"optimizer_results_{timestamp}" if args.output_dir is None else Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger("Optimizer Pipeline", self.output_dir / "optimization.log")
        self.logger.info("Initializing Optimizer Pipeline...")
        self.logger.info(f"Arguments: {vars(args)}")
        
        # Initialize DRR renderer
        self.logger.info(f"Loading CT volume for patient index {args.index}...")
        self.ct_path = Path(args.data_dir) / f"patient_{args.index:02d}/ct.nii.gz"
        subject = read(volume=str(self.ct_path), orientation="AP", center_volume=True)
        self.drr = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX).to(self.device)
        
        # Initialize Optimizer (which holds PoseRegressor internally)
        self.logger.info("Initializing Pose Regressor Optimizer...")
        self.optimizer = Optimizer(drr=self.ct_path).to(self.device)
        
        # Load weights for PoseRegressor
        self.pose_regressor = PoseRegressor()
        self.pose_regressor = self.pose_regressor.eval()
        self.pid = int(self.args.index)
        self.ckpt_dir = self.args.ckpt_dir
        ckpt_path = Path(self.ckpt_dir) / f"regressor_patient{self.pid:02d}.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(self.ckpt_dir) / "regressor_best.pth"
            if not ckpt_path.exists():
                ckpt_path = Path(self.ckpt_dir) / "regressor_last.pth"
        if ckpt_path.exists():
            self.logger.info(f"Loading PoseRegressor checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model_sd" in ckpt:
                self.pose_regressor.load_state_dict(ckpt["model_sd"])
            elif "model" in ckpt:
                self.pose_regressor.load_state_dict(ckpt["model"])
            else:
                self.pose_regressor.load_state_dict(ckpt)
        else:
            self.logger.warning("No valid PoseRegressor checkpoint provided. Using untrained weights!")

    def generate_random_pose(self):
        """
        Generates a valid random GT pose to be used for rendering.
        
        Returns:
            torch.Tensor: (1, 6) tensor containing Euler angles (Z,X,Y) and translations.
        """
        # Euler angles (Z,X,Y) in radians
        # angle_stds = torch.tensor([0.3, 0.3, 0.3], device=self.device)
        angle_stds = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        euler_angles = torch.randn(3, device=self.device) * angle_stds
        
        # Translations in mm
        # trans_stds = torch.tensor([40.0, 50.0, 40.0], device=self.device)
        trans_stds = torch.tensor([10.0, 10.0, 20.0], device=self.device)
        translation = torch.randn(3, device=self.device) * trans_stds
        translation[1] += 650.0  # Base y-offset typically used in the dataset
        
        return torch.cat([euler_angles, translation], dim=-1).unsqueeze(0)

    def compute_metrics(self, gt_pose, pred_pose):
        """
        Computes geodesic rotation error and L2 translation error between ground truth and prediction.
        
        Args:
            gt_pose (torch.Tensor): Ground truth pose (1, 6)
            pred_pose (torch.Tensor): Predicted pose (1, 6)
            
        Returns:
            tuple: (rot_err_deg, trans_err_mm)
        """
        if gt_pose.shape[1] == 6:
            gt_rot = gt_pose[:, :3]
            gt_trans = gt_pose[:, 3:]
            gt_matrix = euler_angles_to_matrix(gt_rot, convention="ZXY")
        elif gt_pose.shape[1] == 9:
            gt_rot = gt_pose[:, :6]
            gt_trans = gt_pose[:, 6:]
            gt_matrix = rotation_6d_to_matrix(gt_rot)

        if pred_pose.shape[1] == 6:
            pred_rot = pred_pose[:, :3]
            pred_trans = pred_pose[:, 3:]
            pred_matrix = euler_angles_to_matrix(pred_rot, convention="ZXY")
        elif pred_pose.shape[1] == 9:
            pred_rot = pred_pose[:, :6]
            pred_trans = pred_pose[:, 6:]
            pred_matrix = rotation_6d_to_matrix(pred_rot)
        
        rot_dist = compute_geodesic_distance(pred_matrix, gt_matrix)
        rot_err_deg = torch.rad2deg(rot_dist).item()
        
        trans_err_mm = torch.norm(pred_trans - gt_trans, dim=1).item()
        
        return rot_err_deg, trans_err_mm

    def compute_loss(self, gt_image, gt_pose, rot_6d_pred, trans_pred):
            euler_gt = gt_pose[:, :3]
            trans_gt = gt_pose[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            
            trans_scale = torch.tensor([40, 50, 40], device=config.DEVICE)
            loss_trans_by_axis = nn.functional.smooth_l1_loss(trans_pred, trans_gt, reduction='none')  / trans_scale
            loss_trans = loss_trans_by_axis.mean()

            trans_weight = trans_scale = torch.tensor(0.5 * [1.0, 0.2, 1.0], device=config.DEVICE)
            loss = loss_rot + (trans_weight * loss_trans_by_axis).mean()

            return loss, loss_rot, loss_trans

    def save_visualization(self, idx, gt_carm, init_proj, opt_proj, init_gain, final_gain):
        """
        Saves a visualization showing the ground truth C-arm image alongside the 
        initial prediction projection and the optimized projection.
        """
        cols = 3
        fig, axes = plt.subplots(1, cols, figsize=(15, 5))
        
        gt_img = gt_carm[0].squeeze().cpu().numpy()
        init_img = init_proj[0].squeeze().cpu().numpy()
        opt_img = opt_proj[0].squeeze().cpu().numpy()
        
        images = [gt_img, init_img, opt_img]
        titles = [
            "Ground Truth C-arm", 
            f"Init Proj (Gain: {init_gain:.3f})", 
            f"Opt Proj (Gain: {final_gain:.3f})"
        ]
        
        for j, (im, t) in enumerate(zip(images, titles)):
            axes[j].imshow(im, cmap="gray")
            axes[j].set_title(t)
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"sample_{idx:03d}.png", bbox_inches='tight')
        plt.close()

    def find_good_pose(self, max_loss = 10.0):
        for i in range(100):
            gt_pose = self.generate_random_pose()
            projection = self.optimizer.render_drr(gt_pose)
            rot_6d_pred, trans_pred = self.optimizer.pose_regressor(projection)
                    
            euler_gt = gt_pose[:, :3]
            trans_gt = gt_pose[:, 3:]
            rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

            loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt).mean()
            trans_scale = 50.0
            trans_weight = 0.5
            loss_trans = nn.functional.smooth_l1_loss(trans_pred / trans_scale, trans_gt / trans_scale)

            loss = loss_rot + trans_weight * loss_trans

            if loss < max_loss:
                break
        print(f"GT pose loss was {loss}")
        return gt_pose

    def run(self, input_img):
        """
        Main execution loop. Generates runtime renders, optimizes poses, and aggregates metrics.
        """
        num_samples = self.args.num_samples
        self.logger.info(f"Starting execution for {num_samples} dynamic samples.")
        
        metrics = {
            "init_rot_err": [], "init_trans_err": [],
            "opt_rot_err": [], "opt_trans_err": [],
            "init_gain": [], "opt_gain": []
        }
    
        # Estimate initial pose
        rot_6d_pred, trans_pred = self.pose_regressor(input_img)

        # Convert to Euler angles
        rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
        rot_euler_pred = matrix_to_euler_angles(rot_matrix_pred)
        initial_pose = torch.cat([rot_euler_pred, trans_pred], dim=-1)

        # Optimize
        best_pose, best_gain, history = self.optimizer(carm=input_img, initial_pose=initial_pose, iters=self.args.iters, patience=self.args.patience)
        
        # 3. Extract Initial Gain
        init_gain = history['gain'][0]
                
        # 5. Log Results
        self.logger.info(
            f"Initial -> Gain: {init_gain:.4f} | Rotation: {torch.rad2deg(rot_euler_pred)} deg | Translation: {trans_pred} mm"
        )
        self.logger.info(
            f"Final   -> Gain: {best_gain:.4f} | Rotation: {torch.rad2deg(best_pose[:, :3])} deg | Translation: {best_pose[:, 3:]} mm"
        )
        
        # 7. Visualization
        if i < self.args.num_visualize:
            with torch.no_grad():
                init_proj = self.optimizer.render_drr(init_pose, parameterization="rotation_6d")
            self.save_visualization(
                idx=i+1, 
                gt_carm=carm, 
                init_proj=init_proj, 
                opt_proj=best_proj,
                init_gain=init_gain, 
                final_gain=final_gain
            )
                
        # # Aggregate statistics and summarize
        # self.logger.info("=== Aggregated Metrics ===")
        # self.logger.info(f"Mean Initial Rot Err: {np.mean(metrics['init_rot_err']):.2f} +/- {np.std(metrics['init_rot_err']):.2f} deg")
        # self.logger.info(f"Mean Optimized Rot Err: {np.mean(metrics['opt_rot_err']):.2f} +/- {np.std(metrics['opt_rot_err']):.2f} deg")
        # self.logger.info(f"Mean Initial Trans Err: {np.mean(metrics['init_trans_err']):.2f} +/- {np.std(metrics['init_trans_err']):.2f} mm")
        # self.logger.info(f"Mean Optimized Trans Err: {np.mean(metrics['opt_trans_err']):.2f} +/- {np.std(metrics['opt_trans_err']):.2f} mm")
        # self.logger.info(f"Mean Initial Gain: {np.mean(metrics['init_gain']):.4f}")
        # self.logger.info(f"Mean Optimized Gain: {np.mean(metrics['opt_gain']):.4f}")
        self.logger.info("Execution complete.")


if __name__ == "__main__":
    # Ensure reproducible pseudo-random behavior optionally
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    args = parse_args()
    pipeline = RunFullPipeline(args)
    pipeline.run()
