import os
import argparse
import random
import csv
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from src.utils import config as config
from src.core.encoder import XrayEncoder
from src.core.estimator import PositionEstimator
from src.core.registration import LatentPoseOptimizer, PoseOptimizer, PoseGenerator, PoseRegressorOptimizer
from src.data.dataset import PoseDataset, DRRMetadataDataset
from src.utils import image_processing
from src.utils.loss import PositionLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Run pose inference pipeline")
    parser.add_argument("--index", type=int, default=3, help="Patient index ID")
    parser.add_argument("--crm_path", type=str, default=None, help="Path to CRM image")
    parser.add_argument("--ct_path", type=str, default=None, help="Path to CT volume")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Encoder/Estimator checkpoint directory")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--pipeline", type=str, default="regressor", choices=["estimator", "regressor"], 
                        help="Which pipeline to run: 'estimator' or 'regressor'")
    parser.add_argument("--regressor_ckpt", type=str, default=None, help="PoseRegressor checkpoint path")
    parser.add_argument("--iterations", type=int, default=50, help="Number of loop iterations")
    return parser.parse_args()


class InferencePipeline:
    def __init__(self, patient_index: int = 2, crm_image_path: str = None, ct_volume_path: str = None, 
                 output_dir: str = None, ckpt_dir: str = None, data_dir: str = None):
        if crm_image_path is None:
            self.crm_image_path = str(config.CRM_DIR / f"{patient_index}.png")
        else:
            self.crm_image_path = crm_image_path
            
        if ct_volume_path is None:
            self.ct_volume_path = str(config.CT_DIR / f"{patient_index}.nii.gz")
        else:
            self.ct_volume_path = ct_volume_path
            
        if output_dir is None:
            self.output_dir = str(config.OUTPUT_DIR / f"{patient_index}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = config.DEVICE
        self.patient_index = patient_index
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.data_dir = data_dir
        
        print(f"Initializing InferencePipeline for Patient {patient_index} on {self.device}")
        
        self.encoder = None
        self.position_estimator = None
        self.pose_optimizer = None
        self.loss_fn = PositionLoss().to(self.device)
        self.crm = None
        
        self._load_pose_ranges()

    def _load_pose_ranges(self, json_filename="metadata.json"):
        """
        Compute per-axis min/max ranges for rotations and translations from a JSON file.
        """
        if self.data_dir is None:
            self.rot_range = None
            self.trans_range = None
            return
            
        json_path = os.path.join(self.data_dir, json_filename)
        if not os.path.exists(json_path):
            self.rot_range = None
            self.trans_range = None
            return
            
        with open(json_path, "r") as f:
            data = json.load(f)

        poses = np.array([item["pose"] for item in data.values()], dtype=np.float64)
        rotations = poses[:, :3]
        translations = poses[:, 3:]

        rot_min, rot_max = rotations.min(axis=0), rotations.max(axis=0)
        trans_min, trans_max = translations.min(axis=0), translations.max(axis=0)

        self.rot_range = torch.tensor(np.stack((rot_min, rot_max), axis=1), device=self.device)
        self.trans_range = torch.tensor(np.stack((trans_min, trans_max), axis=1), device=self.device)

    @torch.no_grad()
    def load_crm(self, crm_path, flip=True):
        import torchvision.io as io
        crm = (io.read_image(crm_path).float().to(self.device) / 255.0).unsqueeze(0)
        if flip:
            crm = crm.flip(dims=[3])
        crm = F.interpolate(crm, size=(config.IMAGE_SIZE, config.IMAGE_SIZE), mode='bilinear', align_corners=False)
        return crm

    def init_estimator_pipeline(self):
        """Initializes XrayEncoder and PositionEstimator (LatentPoseOptimizer uses this)"""
        print("Loading XrayEncoder and PositionEstimator models...")
        self.encoder = XrayEncoder(
            device=self.device,
            size=config.IMAGE_SIZE,
            patch_size=config.PATCH_SIZE,
        ).to(self.device)

        self.position_estimator = PositionEstimator(
            encoder=self.encoder,
            dicom_file=self.ct_volume_path,
            crm_path=self.crm_image_path,
        ).to(self.device)

        if self.ckpt_dir:
            estimator_ckpt_path = self.ckpt_dir / f"estimator{self.patient_index}_best.pth"
            if os.path.exists(estimator_ckpt_path):
                print(f"Loading checkpoint from {estimator_ckpt_path}")
                ckpt = torch.load(estimator_ckpt_path, map_location=self.device, weights_only=False)
                new_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in ckpt["model_sd"].items()})
                self.position_estimator.load_state_dict(new_state_dict, strict=True)
            else:
                print(f"Warning: Checkpoint {estimator_ckpt_path} not found. Running with random/init weights.")
                
        self.crm = self.position_estimator.load_crm(self.crm_image_path)

    def init_regressor_pipeline(self, regressor_ckpt_path=None):
        """Initializes models required for the PoseRegressorOptimizer"""
        print("Initializing PoseRegressorOptimizer...")
        
        from diffdrr.drr import DRR
        from diffdrr.data import read
        
        subject = read(volume=str(self.ct_volume_path), orientation="AP", center_volume=True)
        self.drr = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX).to(self.device)
        self.crm = self.load_crm(self.crm_image_path)
        
        self.pose_optimizer = PoseRegressorOptimizer(drr=self.drr).to(self.device)
        
        if regressor_ckpt_path and os.path.exists(regressor_ckpt_path):
            print(f"Loading PoseRegressor checkpoint from {regressor_ckpt_path}")
            ckpt = torch.load(regressor_ckpt_path, map_location=self.device, weights_only=False)
            # Adjust if model state dict is nested
            if "model_sd" in ckpt:
                self.pose_optimizer.pose_regressor.load_state_dict(ckpt["model_sd"])
            elif "model" in ckpt:
                self.pose_optimizer.pose_regressor.load_state_dict(ckpt["model"])
            else:
                self.pose_optimizer.pose_regressor.load_state_dict(ckpt)
        else:
            print("Warning: PoseRegressor checkpoint not provided or not found.")
            
    def run_estimator_pipeline(self, num_iterations=50):
        print("Starting Estimator Pipeline Inference...")
        self.init_estimator_pipeline()
        self.pose_optimizer = LatentPoseOptimizer(position_estimator=self.position_estimator).to(self.device)
        self._run_optimization_loop(num_iterations)

    def run_regressor_pipeline(self, num_iterations=50, regressor_ckpt_path=None):
        print("Starting Regressor Pipeline Inference...")
        self.init_regressor_pipeline(regressor_ckpt_path)
        self._run_optimization_loop(num_iterations)

    def _run_optimization_loop(self, num_iterations):
        print(f"Starting loop for {num_iterations} iterations...")
        
        init_gains, final_gains = [], []
        init_feat_mse, final_feat_mse = [], []
        init_feat_sim, final_feat_sim = [], []
        init_pos_loss, final_pos_loss = [], []
        steps_log = []

        for j_iter in range(num_iterations):
            print(f'-- Iteration {j_iter} --')
            
            # Generate GT pose for testing
            angle_stds = torch.tensor([0.3, 0.3, 0.3], device=self.device)
            rotation = torch.randn(3, device=self.device) * angle_stds
            trans_stds = torch.tensor([40, 50, 40], device=self.device)
            translation = torch.randn(3, device=self.device) * trans_stds
            translation[1] += 650
            
            gt_pose = torch.cat([rotation, translation], dim=-1).unsqueeze(0).to(self.device)
            
            # Generate synthetic target CRM
            if hasattr(self, 'position_estimator') and self.position_estimator is not None:
                new_crm = self.position_estimator.project(gt_pose)
            else:
                new_crm = self.pose_optimizer.render_drr(gt_pose)
                
            if hasattr(self.pose_optimizer, "update_carm"):
                self.pose_optimizer.update_carm(new_crm)
            else:
                self.pose_optimizer.update_crm(new_crm)
            
            # Optimize
            best_pose, projection_optimized, step, init_results, final_results = self.pose_optimizer()
            
            init_gain, init_position, init_features_mse_loss, init_features_cos_sim = init_results
            final_gain, final_position, final_features_mse_loss, final_features_cos_sim = final_results

            with torch.no_grad():
                if hasattr(self, 'position_estimator') and self.position_estimator is not None:
                    init_proj = self.position_estimator.project(init_position)
                else:
                    init_proj = self.pose_optimizer.render_drr(init_position)

            steps_log.append(step)
            init_gains.append(init_gain)
            final_gains.append(final_gain)
            init_feat_mse.append(init_features_mse_loss)
            final_feat_mse.append(final_features_mse_loss)
            init_feat_sim.append(init_features_cos_sim)
            final_feat_sim.append(final_features_cos_sim)
            
            init_pos_loss.append(self.loss_fn(init_position, gt_pose).item())
            final_pos_loss.append(self.loss_fn(final_position, gt_pose).item())
            
            self._save_pose_csv(j_iter, init_position, final_position, gt_pose)
            self._save_visualization(
                j_iter, new_crm, init_proj, projection_optimized,
                init_gain, final_gain
            )

        self._save_summary_plots(
            steps_log, init_gains, final_gains, 
            init_pos_loss, final_pos_loss, init_feat_mse, final_feat_mse,
            init_feat_sim, final_feat_sim
        )
        print("Done.")

    def fit_mlp(self, model, initial_pose, lr=1e-3, latent_dim=32, patience=100, thr=1e-4):
        """
        Standalone helper retained from original code to fit an MLP to a pose.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss().to(self.device)

        z = torch.randn(1, latent_dim, device=self.device)
        z.requires_grad = False
        
        no_improve  = 0
        best_loss = torch.inf

        for step in range(1000):
            optimizer.zero_grad()
            raw_output = model(z)
            loss = criterion(initial_pose, raw_output)
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1

            if (no_improve > patience) or (best_loss < thr):
                print(f"Stop at step {step} (patience={patience})")
                break

        return model, z, best_loss

    def _save_visualization(self, idx, gt_crm, init_proj, opt_proj, init_gain, final_gain):
        cols = 3
        plt.figure(figsize=(12, 6))
        
        kernel = 1
        if hasattr(self, 'position_estimator') and self.position_estimator is not None:
            kernel = self.position_estimator.kernel
        
        gt_img = gt_crm[0].squeeze().cpu().numpy()
        init_img = (init_proj * kernel)[0].squeeze().cpu().numpy()
        opt_img = (opt_proj * kernel)[0].squeeze().cpu().numpy()
        
        with torch.no_grad():
            try:
                sobel_module = self.pose_optimizer.sobel
            except AttributeError:
                from src.core.layers import Sobel
                sobel_module = Sobel().to(gt_crm.device)
                
            sobel_gt = (sobel_module(gt_crm) * kernel)[0].squeeze().cpu().numpy()
            sobel_init = (sobel_module(init_proj) * kernel)[0].squeeze().cpu().numpy()
            sobel_opt = (sobel_module(opt_proj) * kernel)[0].squeeze().cpu().numpy()

        imgs = [gt_img, init_img, opt_img]
        titles = ["GT", f"Init: {init_gain:.3f}", f"Opt: {final_gain:.3f}"]
        
        for j, (im, t) in enumerate(zip(imgs, titles)):
            plt.subplot(2, cols, j + 1)
            plt.imshow(im, cmap="gray")
            plt.title(t)
            plt.axis("off")

        sobels = [sobel_gt, sobel_init, sobel_opt]
        stitles = ["GT Sobel", "Init Sobel", "Opt Sobel"]
        for j, (im, t) in enumerate(zip(sobels, stitles)):
            plt.subplot(2, cols, cols + j + 1)
            plt.imshow(im, cmap="gray")
            plt.title(t)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pred_{idx}.png", bbox_inches='tight')
        plt.close()

    def _save_summary_plots(self, steps, i_gains, f_gains, i_ploss, f_ploss, i_fmse, f_fmse, i_fsim, f_fsim):
        def plot_metric(data_list, title, ylabel, filename, legends=None):
            plt.figure(figsize=(10, 4))
            if isinstance(data_list, list) and isinstance(data_list[0], list): 
                 for data, leg in zip(data_list, legends):
                     plt.plot(data, label=leg)
                 plt.legend()
            else:
                 plt.plot(data_list)
            plt.title(title); plt.xlabel("Iter"); plt.ylabel(ylabel); plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{filename}")
            plt.close()

        plot_metric(steps, "Steps to Converge", "Steps", "steps.png")
        plot_metric([i_gains, f_gains], "MNCC Gain", "Gain", "gains.png", ["Init", "Final"])
        plot_metric([i_ploss, f_ploss], "Position Loss", "MSE", "pos_loss.png", ["Init", "Final"])
        plot_metric([i_fmse, f_fmse], "Feature MSE", "MSE", "feat_mse.png", ["Init", "Final"])
        plot_metric([i_fsim, f_fsim], "Feature Cos Sim", "Sim", "feat_sim.png", ["Init", "Final"])

    def _save_pose_csv(self, iteration, init_pose, final_pose, gt_pose=None):
        filepath = os.path.join(self.output_dir, "poses.csv")
        init_pose = init_pose.detach().cpu().view(-1).numpy()
        final_pose = final_pose.detach().cpu().view(-1).numpy()

        if gt_pose is not None:
            gt_pose = gt_pose.detach().cpu().view(-1).numpy()

        mode = 'w' if iteration == 0 else 'a'

        with open(filepath, mode=mode, newline="") as f:
            writer = csv.writer(f)
            if iteration == 0:
                writer.writerow(["iteration", "type", "yaw_z", "roll_x", "pitch_y", "tx", "ty", "tz"])
            if gt_pose is not None:
                writer.writerow([iteration, "ground_truth", *gt_pose])
            writer.writerow([iteration, "initial", *init_pose])
            writer.writerow([iteration, "optimized", *final_pose])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if __name__ == "__main__":
    set_seed()
    args = parse_args()

    pipeline = InferencePipeline(
        patient_index=args.index,
        crm_image_path=args.crm_path,
        ct_volume_path=args.ct_path,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
    )
    
    if args.pipeline == "estimator":
        pipeline.run_estimator_pipeline(num_iterations=args.iterations)
    elif args.pipeline == "regressor":
        pipeline.run_regressor_pipeline(num_iterations=args.iterations, regressor_ckpt_path=args.regressor_ckpt)
