import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import datetime
from pathlib import Path
import re, json, time, traceback

import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle

from src.utils import config
from src.utils.training import setup_logger
from src.utils.loss import compute_geodesic_distance, LocalNCC, mGNCCLoss
from src.core.registration import PoseRegressorOptimizer, Optimizer
from src.core.pose_regressor import PoseRegressor
from src.core.multiViewOptimizer import SingleViewOpt, MultiViewOptimizer, ViewData
from src.utils.pose_utils import compute_relative_transform, compose_poses

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
    parser.add_argument("--input", type=str, default=None, help="Path to input C-arm image")
    
    # Optimizer settings
    parser.add_argument("--iters", type=int, default=250, help="Max optimization iterations per sample")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping during optimization")
    
    # Execution settings
    parser.add_argument("--num_samples",    type=int,               default=50, help="Number of random poses/images to evaluate")
    parser.add_argument("--num_visualize",  type=int,               default=5,  help="Number of samples to visualize and save")
    parser.add_argument("--test_synthetic", action="store_true",                help="Run full pipeline using synthetic CT to validate optimization")
    parser.add_argument("--exp1",           action="store_true",                help="Run Experiment over all testset patients")
    parser.add_argument("--exp2",           action="store_true",                help="Run Experiment over all testset patients")
    parser.add_argument("--exp_type",       default='exp_joint',    type=str,   help="Path to input C-arm image")
    
    return parser.parse_args()






class PipelineExperiment:
    """
    Self-contained pipeline class for multi-patient benchmarking experiments.

    Supports two experiment modes selected via CLI flags:
      - ``--exp1``: Joint multi-view optimization using MultiViewOptimizer.
      - ``--exp2``: Single-view optimization with per-patient regressor checkpoints.

    Usage
    ─────
        exp = PipelineExperiment(args)
        exp.exp1()

    Output layout
    ─────────────
    <output_dir>/exp1_<timestamp>/
        patient_<XX>/
            run_000/output_image.png
            run_001/output_image.png
            ...
            patient_report.json
            patient_summary.txt
        overall_report.json
        overall_summary.txt
    """

    # ------------------------------------------------------------------
    # Metric keys tracked for every run
    # ------------------------------------------------------------------
    METRIC_KEYS = [
        "init_simlarity", "final_gain", "gain_delta",
        "init_rot_err_deg", "final_rot_err_deg", "rot_improvement_deg",
        "init_trans_err_mm", "final_trans_err_mm", "trans_improvement_mm",
        "converge_iter", "total_iters",
        "run_time_s",
    ]

    def __init__(self, args):
        self.args = args
        self.device = config.DEVICE

        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.OUTPUT_DIR) / f"exp1_{timestamp}" if args.output_dir is None else Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("Optimizer Pipeline", self.output_dir / "optimization.log")
        self.logger.info("Initializing Optimizer Pipeline...")
        self.logger.info(f"Arguments: {vars(args)}")

        # Initialize DRR renderer
        self.logger.info(f"Loading CT volume for patient index {args.index}...")
        self.ct_path = Path(args.data_dir) / f"patient_{args.index:02d}/ct.nii.gz"
        subject = read(volume=str(self.ct_path), orientation="AP", center_volume=True)
        self.drr = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX).to(self.device)
        
        # Initialize Optimizer
        self.logger.info("Initializing MultiViewOptimizer...")
        self.optimizer = MultiViewOptimizer(ct_path=self.ct_path, loss=mGNCCLoss(), scales=3, lr=[0.001, 1]).to(self.device)

        self.load_checkpoints(id=int(self.args.index))

        self.create_experiments_registry()

    def create_experiments_registry(self):
        self.registry = {
            "exp_joint": self.exp1,
            "exp_sequntial": self.exp_sequntial_optimization,
        }

    def load_checkpoints(self, id:int=None):
        # Load weights for PoseRegressor
        self.pose_regressor = PoseRegressor()
        self.pose_regressor = self.pose_regressor.eval()
        self.pid = int(self.args.index) if id is None else id
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
            self.pose_regressor.to(self.device)
        else:
            self.logger.warning("No valid PoseRegressor checkpoint provided. Using untrained weights!")

    def pose_to_matrix(self, pose):
        """
        Converts pose tensor into 4x4 transform matrix.

        pose should be a tensor shape (N, 6)
        """
        R = euler_angles_to_matrix(pose[:, :3], convention="ZXY")

        t = pose[..., 3:]

        T = torch.eye(4, device=self.device)

        T[:3, :3] = R.squeeze()
        T[:3, 3] = t.squeeze()

        return T

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

    def compute_pde(self, gt_pose, pred_pose, points=None):
        """
        Pose-induced displacement error (mm).
        """
        if points is None:
            points = torch.tensor([
                [-50, -50, -50],
                [-50, -50,  50],
                [-50,  50, -50],
                [-50,  50,  50],
                [ 50, -50, -50],
                [ 50, -50,  50],
                [ 50,  50, -50],
                [ 50,  50,  50],
                [  0,   0,   0],
            ], dtype=torch.float32, device=self.device)

        T_gt = self.pose_to_matrix(gt_pose)
        T_pr = self.pose_to_matrix(pred_pose)

        ones = torch.ones((points.shape[0], 1), device=self.device)

        pts_h = torch.cat([points, ones], dim=1)

        gt_pts = (T_gt @ pts_h.T).T[:, :3]
        pr_pts = (T_pr @ pts_h.T).T[:, :3]

        err = torch.linalg.norm(gt_pts - pr_pts, dim=1)

        return err.mean().item()

    def save_visualization(self, gt_img, init_proj, opt_proj, init_simlarity, final_gain, filename=None):
        """
        Saves a visualization showing the ground truth C-arm image alongside the 
        initial prediction projection and the optimized projection.
        """
        save_path = self.output_dir / "compare_results" if filename is None else self.output_dir / filename
        cols = 3
        fig, axes = plt.subplots(1, cols, figsize=(15, 5))
        
        gt_img = gt_img[0].squeeze().cpu().numpy()
        init_img = init_proj[0].squeeze().detach().cpu().numpy()
        opt_img = opt_proj[0].squeeze().cpu().numpy()
        
        images = [gt_img, init_img, opt_img]
        titles = [
            "Ground Truth C-arm", 
            f"Init Proj (Gain: {init_simlarity:.3f})", 
            f"Opt Proj (Gain: {final_gain:.3f})"
        ]
        
        for j, (im, t) in enumerate(zip(images, titles)):
            axes[j].imshow(im, cmap="gray")
            axes[j].set_title(t)
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def generate_random_pose(self, N=1, seed=13):
        """
        Generates a valid random GT pose to be used for rendering.
        
        Returns:
            torch.Tensor: (N, 2, 6) tensor containing Euler angles (Z,X,Y) and translations for 2 views.
        """
        # Set fixed seed for determinstic experiment
        torch.manual_seed(seed)

        # Euler angles (Z,X,Y) in radians
        angle_stds = torch.tensor([0.3, 0.3, 0.3], device=self.device)
        euler_angles = torch.randn((N, 2, 3), device=self.device) * angle_stds
        
        # Translations in mm
        trans_stds = torch.tensor([20.0, 80.0, 20.0], device=self.device)
        translation = torch.randn((N, 2, 3), device=self.device) * trans_stds
        translation[..., 1] += 580.0  # Base y-offset typically used in the dataset
        
        gt_poses = torch.cat([euler_angles, translation], dim=-1)
        return gt_poses

    def find_good_pose(self, N=1, max_loss = 10.0):
        poses_out = torch.zeros((N, 2, 6))
        for step in range(2):
            for n in range(N):
                for i in range(100):
                    gt_poses = self.generate_random_pose(N=1, seed=np.random.randint(1000))
                    gt_pose = gt_poses[0, 0, :].unsqueeze(0)
                    
                    euler_gt = gt_pose[:, :3]
                    trans_gt = gt_pose[:, 3:]
                    rot_matrix_gt = euler_angles_to_matrix(euler_gt, convention="ZXY")

                    projection = self.optimizer.render_drr(rot_matrix_gt, trans_gt, self.drr)
                    rot_6d_pred, trans_pred = self.pose_regressor(projection)

                    rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)

                    loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt)
                    trans_scale = 50.0
                    trans_weight = 0.5
                    loss_trans = nn.functional.smooth_l1_loss(trans_pred / trans_scale, trans_gt / trans_scale)

                    loss = loss_rot + trans_weight * loss_trans

                    if loss < max_loss:
                        break

                poses_out[n, step, :] = gt_pose

        return poses_out

    # ------------------------------------------------------------------
    # Patient discovery
    # ------------------------------------------------------------------
    def _discover_patients(self) -> list[Path]:
        """Return sorted list of patient_XX subdirectories in args.data_dir."""
        data_dir = Path(self.args.data_dir)
        return sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and re.match(r"patient_\d+$", p.name)
        )

    # ------------------------------------------------------------------
    # Per-patient initialisation
    # ------------------------------------------------------------------
    def _reinit_patient(self, pat_dir: Path, pat_index: int=None) -> bool:
        """
        Re-initialise self.optimizer and self.drr for a new patient CT.
        Returns True on success, False if the CT is missing or loading fails.
        """
        ct_path = pat_dir / "ct.nii.gz"
        self.ct_path = ct_path
        if not ct_path.exists():
            self.logger.warning(f"  CT not found at {ct_path}, skipping.")
            return False

        try:
            self.optimizer = type(self.optimizer)(
                ct_path = ct_path,
                loss    = self.optimizer.criterion,
                lr      = self.optimizer.lr,
                scales  = len(self.optimizer.scales),
                device  = self.device,
            ).to(self.device)

            subject   = read(volume=str(ct_path), orientation="AP", center_volume=True)
            self.drr  = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX).to(self.device)

        except Exception as exc:
            self.logger.error(f"  Failed to load {pat_dir.name}: {exc}")
            self.logger.error(traceback.format_exc())
            return False

        if self.args.exp2:
            self.load_checkpoints(id=pat_index)

        return True

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, exp:str = 'exp1'):
        """
        For every patient_XX folder found under args.data_dir:
          1. Re-initialise the Optimizer for that patient's CT.
          2. Run the full regressor → optimiser pipeline num_samples times,
             each time with a freshly sampled random GT pose.
          3. Record per-run metrics (gains, errors, timing, convergence).
          4. Save a visualisation image for every run.
          5. Write a per-patient JSON + plain-text summary report.

        After all patients are processed a global summary (JSON + text) is saved.
        """
        num_samples = self.args.num_samples

        self.logger.info(f"\n{'═'*60}")
        self.logger.info( "  EXP1 – multi-patient benchmark")
        self.logger.info(f"  Runs per patient : {num_samples}")
        self.logger.info(f"  Output root      : {self.output_dir}")
        self.logger.info(f"{'═'*60}")

        patient_dirs = self._discover_patients()
        if not patient_dirs:
            self.logger.error(f"No patient_XX folders found under {self.args.data_dir}")
            return

        self.logger.info(f"Found {len(patient_dirs)} patient(s): {[p.name for p in patient_dirs]}")

        all_patients_metrics: dict[str, list[dict]] = {}

        for pat_dir in patient_dirs:
            pat_name   = pat_dir.name
            pat_index  = int(pat_name.split('_')[-1])
            pat_outdir = self.output_dir / pat_name
            pat_outdir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n{'─'*60}")
            self.logger.info(f"  Processing {pat_name}")
            self.logger.info(f"{'─'*60}")

            if not self._reinit_patient(pat_dir, pat_index):
                continue                       # CT missing or failed to load

            # self.gt_poses = self.generate_random_pose(
            #     N    = num_samples, 
            #     seed = pat_index,
            # )
            self.gt_poses = self.find_good_pose(N=num_samples, max_loss=10).to(self.device)
            
            
            if exp not in self.registry.keys():
                raise ValueError(f"Experiment '{exp}' not found. Available experiments: {list(self.registry.keys())}")
            
            experiment_fn = self.registry[exp]

            pat_runs = self._run_patient_experiment(
                pat_name       = pat_name,
                pat_outdir     = pat_outdir,
                num_samples    = num_samples,
                experiment_fn  = experiment_fn,
            )
            all_patients_metrics[pat_name] = pat_runs

            self._save_patient_report(
                pat_name   = pat_name,
                pat_outdir = pat_outdir,
                pat_runs   = pat_runs,
                num_samples= num_samples,
            )

        self._save_global_report(
            exp_root            = self.output_dir,
            all_patients_metrics= all_patients_metrics,
        )
        self.logger.info(f"\n  Experiment complete.  Results in: {self.output_dir}")

    # ------------------------------------------------------------------
    # Repetition loop for one patient
    # ------------------------------------------------------------------
    def _run_patient_experiment(self, pat_name: str, pat_outdir: Path, num_samples: int, experiment_fn=None) -> list[dict]:
        """
        Runs the pipeline num_samples times for the already-loaded patient.
        
        Args:
            experiment_fn: Callable with signature (run_idx, run_outdir) -> dict | None.
                           Defaults to self._run_single for backward compatibility.

        Returns a list of per-run metric dicts (failed runs are omitted).
        """
        if experiment_fn is None:
            experiment_fn = self._run_single

        pat_runs = []

        for run_idx in range(num_samples):
            self.logger.info(f"\n  [{pat_name}] run {run_idx + 1}/{num_samples}")

            metrics = experiment_fn(run_idx=run_idx, run_outdir=pat_outdir)
            if metrics is not None:
                pat_runs.append(metrics)
                self.logger.info(
                    f"    init_simlarity={metrics['init_simlarity']:.4f}  "
                    f"final_gain={metrics['final_gain']:.4f}  "
                    f"Δ={metrics['gain_delta']:+.4f} | "
                    f"rot  {metrics['init_rot_err_deg']:.2f}→ "
                    f"{metrics['final_rot_err_deg']:.2f}°  "
                    f"trans {metrics['init_trans_err_mm']:.1f}→ "
                    f"{metrics['final_trans_err_mm']:.1f}mm | "
                    f"conv@{metrics['converge_iter']}/{metrics['total_iters']}  "
                    f"{metrics['run_time_s']:.1f}s"
                )

        return pat_runs

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------
    def _run_single(self, run_idx: int, run_outdir: Path) -> dict | None:
        """
        Execute one full pipeline pass (sample pose → regressor → optimise).
        Saves a visualisation to run_outdir and returns a metrics dict,
        or None if any step raises an exception.
        """
        # Temporarily redirect save_visualization output to this run's folder
        orig_output_dir = self.output_dir
        self.output_dir = run_outdir
        filename = f"run_{run_idx:03d}"

        t0 = time.perf_counter()
        try:
            # ── Ground-truth pose & DRR ────────────────────────────────────
            gt_pose   = self.gt_poses[run_idx]
            input_img = self.optimizer.render_drr(gt_pose)

            # ── Regressor: initial pose estimate ──────────────────────────
            with torch.no_grad():
                rot_6d_pred, trans_pred = self.pose_regressor(input_img)

            rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
            rot_euler_pred  = matrix_to_euler_angles(rot_matrix_pred, convention="ZXY")
            initial_pose    = torch.cat([rot_euler_pred, trans_pred], dim=-1)

            with torch.no_grad():
                init_proj = self.optimizer.render_drr(initial_pose, self.drr)
                init_simlarity = self.optimizer.criterion(init_proj, input_img).item()

            init_rot_err, init_trans_err = self.compute_metrics(gt_pose, initial_pose)

            # ── Optimiser ─────────────────────────────────────────────────
            best_pose, best_gain, history = self.optimizer(
                carm            = input_img,
                initial_pose    = initial_pose,
                kernel          = 1.0,
                iters_per_scale = self.args.iters,
                verbose         = False,
            )

            final_rot_err, final_trans_err = self.compute_metrics(gt_pose, best_pose)

            # ── Convergence iteration ─────────────────────────────────────
            converge_iter, total_iters = self._find_convergence_iter(history["gain"])

            run_time = time.perf_counter() - t0

            # ── Visualisation ─────────────────────────────────────────────
            optimized_proj = self.optimizer.render_drr(best_pose, self.drr)
            self.save_visualization(
                gt_img     = input_img,
                init_proj  = init_proj,
                opt_proj   = optimized_proj,
                init_simlarity  = init_simlarity,
                final_gain = best_gain,
                filename   = filename,
            )

            return {
                "run"                  : run_idx,
                "init_simlarity"            : round(init_simlarity, 5),
                "final_gain"           : round(best_gain, 5),
                "gain_delta"           : round(best_gain - init_simlarity, 5),
                "init_rot_err_deg"     : round(init_rot_err, 4),
                "final_rot_err_deg"    : round(final_rot_err, 4),
                "rot_improvement_deg"  : round(init_rot_err - final_rot_err, 4),
                "init_trans_err_mm"    : round(init_trans_err, 3),
                "final_trans_err_mm"   : round(final_trans_err, 3),
                "trans_improvement_mm" : round(init_trans_err - final_trans_err, 3),
                "converge_iter"        : converge_iter,
                "total_iters"          : total_iters,
                "run_time_s"           : round(run_time, 2),
                "gt_pose"              : gt_pose.detach().cpu().tolist()[0],
                "initial_pose"         : initial_pose.detach().cpu().tolist()[0],
                "best_pose"            : best_pose.detach().cpu().tolist()[0],
                "pde_mm"               : self.compute_pde(gt_pose.detach().cpu(), best_pose.detach().cpu())
            }

        except Exception as exc:
            self.logger.error(f"    Run {run_idx} failed: {exc}")
            self.logger.error(traceback.format_exc())
            return None

        finally:
            self.output_dir = orig_output_dir

    # ------------------------------------------------------------------
    # Experiment 1: Multi-View Joint Optimization
    # ------------------------------------------------------------------
    def exp1(self, run_idx: int, run_outdir: Path) -> dict | None:
        """
        Execute one multi-view pipeline pass:
          1. Sample two independent GT poses and render a DRR pair.
          2. Run the pose regressor on each image independently.
          3. Compute the relative transform between the two predicted poses.
          4. Build ViewData objects and call MultiViewOptimizer.forward_multiview().
          5. Report metrics w.r.t. view-0 (reference view) for backward
             compatibility with the existing reporting pipeline.

        Saves a visualisation of the reference view to run_outdir.
        Returns a metrics dict (same structure as _run_single), or None on failure.
        """
        orig_output_dir = self.output_dir
        self.output_dir = run_outdir
        filename = f"run_{run_idx:03d}"

        t0 = time.perf_counter()
        try:
            # ── Ground-truth poses & DRRs ─────────────────────────────────
            # self.gt_poses has shape (N, 2, 6); each row is [euler(3), t(3)]
            gt_pose_pair = self.gt_poses[run_idx]          # (2, 6)
            gt_pose1     = gt_pose_pair[0].unsqueeze(0)    # (1, 6)  reference
            gt_pose2     = gt_pose_pair[1].unsqueeze(0)    # (1, 6)  secondary

            # Convert GT Euler poses to (R, t)
            gt_R1 = euler_angles_to_matrix(gt_pose1[:, :3], convention="ZXY")  # (1,3,3)
            gt_t1 = gt_pose1[:, 3:]                                             # (1,3)
            gt_R2 = euler_angles_to_matrix(gt_pose2[:, :3], convention="ZXY")
            gt_t2 = gt_pose2[:, 3:]

            # Render one DRR per view
            img1 = self.optimizer.render_drr(gt_R1, gt_t1)   # (1,1,H,W) normalized
            img2 = self.optimizer.render_drr(gt_R2, gt_t2)

            # ── Pose regressor: initial estimates ─────────────────────────
            with torch.no_grad():
                rot_6d_pred1, trans_pred1 = self.pose_regressor(img1)
                rot_6d_pred2, trans_pred2 = self.pose_regressor(img2)

            # Convert 6D rotation → matrix → Euler (for metrics / ViewData)
            R1_init = rotation_6d_to_matrix(rot_6d_pred1)   # (1,3,3)
            t1_init = trans_pred1                            # (1,3)
            R2_init = rotation_6d_to_matrix(rot_6d_pred2)
            t2_init = trans_pred2

            # ── Compute initial similarity for the reference view ───────────────
            with torch.no_grad():
                init_proj1 = self.optimizer.render_drr(R1_init, t1_init)
                init_simlarity  = self.optimizer.criterion(init_proj1, img1).item()

            # ── Metrics: initial error w.r.t. reference GT ───────────────
            # compute_metrics expects (1,6) Euler tensors
            euler1_init = matrix_to_euler_angles(R1_init, convention="ZXY")
            initial_pose1 = torch.cat([euler1_init, t1_init], dim=-1)   # (1,6)
            init_rot_err, init_trans_err = self.compute_metrics(gt_pose1, initial_pose1)

            # ── Relative transform (from predicted poses, NOT GT) ─────────
            # R_delta, t_delta = compute_relative_transform(R1_init, t1_init, R2_init, t2_init)
            R_delta, t_delta = compute_relative_transform(gt_R1, gt_t1, gt_R2, gt_t2)

            # ── Build ViewData objects ────────────────────────────────────
            view1 = ViewData(
                image      = img1,
                R_init     = R1_init,
                t_init     = t1_init,
                R_delta_1i = torch.eye(3, device=self.device).unsqueeze(0),
                t_delta_1i = torch.zeros(1, 3, device=self.device),
                weight     = 1.0,
            )
            view2 = ViewData(
                image      = img2,
                R_init     = R2_init,
                t_init     = t2_init,
                R_delta_1i = R_delta,
                t_delta_1i = t_delta,
                weight     = 1.0,
            )

            # ── Multi-view joint optimization ─────────────────────────────
            result = self.optimizer.forward_multiview(
                views           = [view1, view2],
                iters_per_scale = self.args.iters,
                verbose         = False,
            )

            # result.R / result.t → optimized reference-view pose
            best_R    = result.R   # (1,3,3)
            best_t    = result.t   # (1,3)
            
            # result.per_view_scores[0] is the final NCC similarity for the
            # reference view — directly comparable to single-view gain.
            final_gain = result.per_view_scores[0] if result.per_view_scores else -best_loss

            # ── Final metrics (reference view) ────────────────────────────
            euler_best = matrix_to_euler_angles(best_R, convention="ZXY")
            best_pose  = torch.cat([euler_best, best_t], dim=-1)   # (1,6)
            final_rot_err, final_trans_err = self.compute_metrics(gt_pose1, best_pose)

            # ── Convergence ───────────────────────────────────────────────
            # forward_multiview records loss (lower = better); negate to reuse
            # _find_convergence_iter which expects gain (higher = better).
            neg_losses = [-l for l in result.history["loss"]]
            converge_iter, total_iters = self._find_convergence_iter(neg_losses)

            run_time = time.perf_counter() - t0

            # ── Visualisation (reference view only) ───────────────────────
            optimized_proj = self.optimizer.render_drr(best_R, best_t)
            self.save_visualization(
                gt_img     = img1,
                init_proj  = init_proj1,
                opt_proj   = optimized_proj,
                init_simlarity  = init_simlarity,
                final_gain = final_gain,
                filename   = filename,
            )

            return {
                "run"                  : run_idx,
                "init_simlarity"            : round(init_simlarity, 5),
                "final_gain"           : round(final_gain, 5),
                "gain_delta"           : round(final_gain - init_simlarity, 5),
                "init_rot_err_deg"     : round(init_rot_err, 4),
                "final_rot_err_deg"    : round(final_rot_err, 4),
                "rot_improvement_deg"  : round(init_rot_err - final_rot_err, 4),
                "init_trans_err_mm"    : round(init_trans_err, 3),
                "final_trans_err_mm"   : round(final_trans_err, 3),
                "trans_improvement_mm" : round(init_trans_err - final_trans_err, 3),
                "converge_iter"        : converge_iter,
                "total_iters"          : total_iters,
                "run_time_s"           : round(run_time, 2),
                "gt_pose"              : gt_pose1.detach().cpu().tolist()[0],
                "initial_pose"         : initial_pose1.detach().cpu().tolist()[0],
                "best_pose"            : best_pose.detach().cpu().tolist()[0],
                "pde_mm"               : self.compute_pde(gt_pose1.detach().cpu(), best_pose.detach().cpu()),
            }

        except Exception as exc:
            self.logger.error(f"    Run {run_idx} (exp1) failed: {exc}")
            self.logger.error(traceback.format_exc())
            return None

        finally:
            self.output_dir = orig_output_dir

    # ------------------------------------------------------------------
    # Experiment 2: Multi-View Sequntial Optimization
    # ------------------------------------------------------------------
    def exp_sequntial_optimization(self, run_idx: int, run_outdir: Path) -> dict | None:
        """
        Execute one multi-view pipeline pass:
          1. Sample two independent GT poses and render a DRR pair.
          2. Run the pose regressor on each image independently.
          3. Compute the relative transform between the two predicted poses.
          4. Build ViewData objects and call MultiViewOptimizer.forward_multiview().
          5. Report metrics w.r.t. view-0 (reference view) for backward
             compatibility with the existing reporting pipeline.

        Saves a visualisation of the reference view to run_outdir.
        Returns a metrics dict (same structure as _run_single), or None on failure.
        """
        orig_output_dir = self.output_dir
        self.output_dir = run_outdir
        filename = f"run_{run_idx:03d}"

        # Create single view optimizer object
        single_view_optimizer = SingleViewOpt(ct_path=self.ct_path)

        t0 = time.perf_counter()
        try:
            # ── Ground-truth poses & DRRs ─────────────────────────────────
            # self.gt_poses has shape (N, 2, 6); each row is [euler(3), t(3)]
            gt_pose_pair = self.gt_poses[run_idx]          # (2, 6)
            gt_pose1     = gt_pose_pair[0].unsqueeze(0)    # (1, 6)  reference
            gt_pose2     = gt_pose_pair[1].unsqueeze(0)    # (1, 6)  secondary

            # Convert GT Euler poses to (R, t)
            gt_R1 = euler_angles_to_matrix(gt_pose1[:, :3], convention="ZXY")  # (1,3,3)
            gt_t1 = gt_pose1[:, 3:]                                             # (1,3)
            gt_R2 = euler_angles_to_matrix(gt_pose2[:, :3], convention="ZXY")
            gt_t2 = gt_pose2[:, 3:]

            # Render one DRR per view
            img1 = self.optimizer.render_drr(gt_R1, gt_t1)   # (1,1,H,W) normalized
            img2 = self.optimizer.render_drr(gt_R2, gt_t2)

            # ── Pose regressor: initial estimates ─────────────────────────
            with torch.no_grad():
                rot_6d_pred1, trans_pred1 = self.pose_regressor(img1)
                rot_6d_pred2, trans_pred2 = self.pose_regressor(img2)

            # Convert 6D rotation → matrix → Euler (for metrics / ViewData)
            R1_init = rotation_6d_to_matrix(rot_6d_pred1)   # (1,3,3)
            t1_init = trans_pred1                            # (1,3)
            R2_init = rotation_6d_to_matrix(rot_6d_pred2)
            t2_init = trans_pred2

            # ── Compute initial similarity for the reference view ───────────────
            with torch.no_grad():
                init_proj1 = self.optimizer.render_drr(R1_init, t1_init)
                init_simlarity  = self.optimizer.criterion(init_proj1, img1).item()

            # ── Metrics: initial error w.r.t. reference GT ───────────────
            # compute_metrics expects (1,6) Euler tensors
            euler1_init = matrix_to_euler_angles(R1_init, convention="ZXY")
            initial_pose1 = torch.cat([euler1_init, t1_init], dim=-1)   # (1,6)
            init_rot_err, init_trans_err = self.compute_metrics(gt_pose1, initial_pose1)

            # ── Relative transform (from GT poses) ─────────
            R_delta, t_delta = compute_relative_transform(gt_R1, gt_t1, gt_R2, gt_t2)

            # ── Build ViewData objects ────────────────────────────────────
            view1 = ViewData(image = img1, R_init = R1_init, t_init = t1_init, R_delta_1i = torch.eye(3, device=self.device).unsqueeze(0), t_delta_1i = torch.zeros(1, 3, device=self.device), weight= 1.0)
            view2 = ViewData(image = img2, R_init = R2_init, t_init = t2_init, R_delta_1i = R_delta, t_delta_1i = t_delta, weight = 1.0)

            # ── Single-view optimization on reference view ─────────────────────────────
            result_single = single_view_optimizer(carm = img1, R_init = R1_init, t_init = t1_init, iters_per_scale=self.args.iters, patience=self.args.patience, verbose=False)

            # Update reference view with optimized pose
            view1.R_init = result_single.R
            view1.t_init = result_single.t

            # ── Multi-view joint optimization ─────────────────────────────
            result = self.optimizer.forward_multiview(views = [view1, view2], iters_per_scale = self.args.iters, verbose = False)

            # result.R / result.t → optimized reference-view pose
            best_R    = result.R   # (1,3,3)
            best_t    = result.t   # (1,3)
            
            # result.per_view_scores[0] is the final NCC similarity for the
            # reference view — directly comparable to single-view gain.
            final_gain = result.per_view_scores[0] if result.per_view_scores else -best_loss

            # ── Final metrics (reference view) ────────────────────────────
            euler_best = matrix_to_euler_angles(best_R, convention="ZXY")
            best_pose  = torch.cat([euler_best, best_t], dim=-1)   # (1,6)
            final_rot_err, final_trans_err = self.compute_metrics(gt_pose1, best_pose)

            # ── Convergence ───────────────────────────────────────────────
            # forward_multiview records loss (lower = better); negate to reuse
            # _find_convergence_iter which expects gain (higher = better).
            neg_losses = [-l for l in result.history["loss"]]
            converge_iter, total_iters = self._find_convergence_iter(neg_losses)

            run_time = time.perf_counter() - t0

            # ── Visualisation (reference view only) ───────────────────────
            optimized_proj = self.optimizer.render_drr(best_R, best_t)
            self.save_visualization(
                gt_img     = img1,
                init_proj  = init_proj1,
                opt_proj   = optimized_proj,
                init_simlarity  = init_simlarity,
                final_gain = final_gain,
                filename   = filename,
            )

            return {
                "run"                  : run_idx,
                "init_simlarity"            : round(init_simlarity, 5),
                "final_gain"           : round(final_gain, 5),
                "gain_delta"           : round(final_gain - init_simlarity, 5),
                "init_rot_err_deg"     : round(init_rot_err, 4),
                "final_rot_err_deg"    : round(final_rot_err, 4),
                "rot_improvement_deg"  : round(init_rot_err - final_rot_err, 4),
                "init_trans_err_mm"    : round(init_trans_err, 3),
                "final_trans_err_mm"   : round(final_trans_err, 3),
                "trans_improvement_mm" : round(init_trans_err - final_trans_err, 3),
                "converge_iter"        : converge_iter,
                "total_iters"          : total_iters,
                "run_time_s"           : round(run_time, 2),
                "gt_pose"              : gt_pose1.detach().cpu().tolist()[0],
                "initial_pose"         : initial_pose1.detach().cpu().tolist()[0],
                "best_pose"            : best_pose.detach().cpu().tolist()[0],
                "pde_mm"               : self.compute_pde(gt_pose1.detach().cpu(), best_pose.detach().cpu()),
            }

        except Exception as exc:
            self.logger.error(f"    Run {run_idx} (exp1) failed: {exc}")
            self.logger.error(traceback.format_exc())
            return None

        finally:
            self.output_dir = orig_output_dir


    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------
    def _find_convergence_iter(self, gains: list[float]) -> tuple[int, int]:
        """
        Return (converge_iter, total_iters) where converge_iter is the last
        iteration at which the best gain was updated (i.e. patience ran out
        after that point).
        """
        total_iters   = len(gains)
        running_best  = -float("inf")
        converge_iter = total_iters
        for i, g in enumerate(gains):
            if g > running_best + 1e-4:
                running_best  = g
                converge_iter = i
        return converge_iter, total_iters

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _compute_stats(self, runs: list[dict]) -> dict:
        """
        For each numeric key in METRIC_KEYS compute mean/std/median/min/max
        across all runs, plus success-rate percentages.
        Returns a flat dict keyed as  metric__statistic.
        """
        stats = {}
        for k in self.METRIC_KEYS:
            vals = [r[k] for r in runs if k in r and r[k] is not None]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            stats[f"{k}__mean"]   = round(float(np.mean(arr)),   4)
            stats[f"{k}__std"]    = round(float(np.std(arr)),     4)
            stats[f"{k}__median"] = round(float(np.median(arr)),  4)
            stats[f"{k}__min"]    = round(float(np.min(arr)),     4)
            stats[f"{k}__max"]    = round(float(np.max(arr)),     4)

        n = len(runs)
        stats["pct_gain_improved"]  = round(100 * sum(
            1 for r in runs if r.get("gain_delta", -1) > 0) / n, 1)
        stats["pct_rot_improved"]   = round(100 * sum(
            1 for r in runs if r.get("rot_improvement_deg", -1) > 0) / n, 1)
        stats["pct_trans_improved"] = round(100 * sum(
            1 for r in runs if r.get("trans_improvement_mm", -1) > 0) / n, 1)
        stats["failure_rate"] = round(100 * sum(
            1 for r in runs if (r["final_trans_err_mm"] > 50 or r["final_rot_err_deg"] > 20)) / n, 2)
        self._compute_success_rates(runs, stats)
        return stats

    def _compute_success_rates(self, runs: list[dict], stats:dict):
        thresholds = [(5, 3), (10, 5), (20, 8)]
        thresholds_mm = [1, 2, 5, 10]
        n = len(runs)

        for t_thresh, r_thresh in thresholds:
            stats[f"SR_{t_thresh}mm_{r_thresh}deg"] = round(100 * sum(
                1 for r in runs if (r["final_trans_err_mm"] < t_thresh and r["final_rot_err_deg"] < r_thresh)) / n, 2)
        
        for th in thresholds_mm:
            stats[f"SR_{th}mm"] = round(100 * sum(
                    1 for r in runs if r["pde_mm"] < th) / len(runs), 2)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _save_patient_report(self, pat_name:    str, pat_outdir:  Path, pat_runs:    list[dict], num_samples: int):
        if not pat_runs:
            self.logger.warning(f"  No successful runs for {pat_name}.")
            return

        stats = self._compute_stats(pat_runs)
        self._log_stats(
            prefix=f"Patient summary — {pat_name}  "
                   f"({len(pat_runs)}/{num_samples} runs OK)",
            stats=stats,
        )

        report = {
            "patient"     : pat_name,
            "num_ok_runs" : len(pat_runs),
            "num_samples" : num_samples,
            "stats"       : stats,
            "runs"        : pat_runs,
        }
        with open(pat_outdir / "patient_report.json", "w") as f:
            json.dump(report, f, indent=2)

        self._write_patient_txt(
            path        = pat_outdir / "patient_summary.txt",
            title       = f"Patient: {pat_name}  |  {len(pat_runs)}/{num_samples} runs",
            stats       = stats,
            run_table   = pat_runs,
        )
        self.logger.info(f"  Reports saved to {pat_outdir}")

    def _save_global_report(
        self,
        exp_root:             Path,
        all_patients_metrics: dict[str, list[dict]],
    ):
        self.logger.info(f"\n{'═'*60}")
        self.logger.info("  EXP1 – global summary")
        self.logger.info(f"{'═'*60}")

        all_runs = [r for runs in all_patients_metrics.values() for r in runs]
        if not all_runs:
            self.logger.warning("No successful runs across any patient.")
            return

        global_stats         = self._compute_stats(all_runs)
        per_patient_condensed = {
            name: self._compute_stats(runs)
            for name, runs in all_patients_metrics.items()
            if runs
        }

        self._log_stats(
            prefix=f"Global  ({len(all_runs)} total runs, "
                   f"{len(per_patient_condensed)} patients)",
            stats=global_stats,
        )

        global_report = {
            "num_patients"       : len(all_patients_metrics),
            "total_runs"         : len(all_runs),
            "global_stats"       : global_stats,
            "per_patient_stats"  : per_patient_condensed,
        }
        with open(exp_root / "overall_report.json", "w") as f:
            json.dump(global_report, f, indent=2)

        self._write_global_txt(
            path                  = exp_root / "overall_summary.txt",
            global_stats          = global_stats,
            per_patient_condensed = per_patient_condensed,
            total_runs            = len(all_runs),
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_stats(self, prefix: str, stats: dict):
        lines = [f"\n{'─'*55}", f"  {prefix}", f"{'─'*55}"]
        for k, v in stats.items():
            lines.append(f"  {k:<40s}: {v:.4f}" if isinstance(v, float)
                         else f"  {k:<40s}: {v}")
        lines.append(f"{'─'*55}")
        self.logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Text-file writers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_stat_line(
        label: str,
        stats: dict,
        key:   str,
        unit:  str = "",
    ) -> str:
        return (
            f"  {label:<30s}"
            f"  mean={stats[f'{key}__mean']:>8.4f}{unit}"
            f"  std={stats[f'{key}__std']:>7.4f}"
            f"  median={stats[f'{key}__median']:>8.4f}"
            f"  [{stats[f'{key}__min']:.4f}, {stats[f'{key}__max']:.4f}]"
        )

    def _write_patient_txt(
        self,
        path:      Path,
        title:     str,
        stats:     dict,
        run_table: list[dict],
    ):
        S = stats
        L = self._fmt_stat_line
        # sr_keys = sorted([k for k in S.keys() if k.startswith("SR")], key=lambda x: (int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else float("inf")))
        sr_keys = sorted([k for k in S.keys() if k.startswith("SR") and "deg" not in k], key=lambda x: float(re.search(r"(\d+\.?\d*)", x).group(1)))
        sr_keys += sorted([k for k in S.keys() if k.startswith("SR") and "deg" in k], key=lambda x: float(re.search(r"(\d+\.?\d*)", x).group(1)))
        sr_lines = [f"  {k:<30s}: {S[k]:>6.2f}%" for k in sr_keys]
                
        lines = [
            "=" * 72,
            f"  {title}",
            "=" * 72,
            "",
            "  GAINS",
            L("  initial gain",      S, "init_simlarity"),
            L("  final gain",        S, "final_gain"),
            L("  gain delta",        S, "gain_delta"),
            f"  Runs where gain improved   : {S['pct_gain_improved']}%",
            "",
            "  ROTATION ERROR",
            L("  initial",           S, "init_rot_err_deg",    "°"),
            L("  final",             S, "final_rot_err_deg",   "°"),
            L("  improvement",       S, "rot_improvement_deg", "°"),
            f"  Runs where rot improved    : {S['pct_rot_improved']}%",
            "",
            "  TRANSLATION ERROR",
            L("  initial",           S, "init_trans_err_mm",    "mm"),
            L("  final",             S, "final_trans_err_mm",   "mm"),
            L("  improvement",       S, "trans_improvement_mm", "mm"),
            f"  Runs where trans improved  : {S['pct_trans_improved']}%",
            "",
            "  SUCCESS RATES",
            *sr_lines,
            "",            
            "  OPTIMISER",
            L("  converge iter",     S, "converge_iter"),
            L("  total iters",       S, "total_iters"),
            L("  run time",          S, "run_time_s", "s"),
            "",
            "  PER-RUN TABLE",
            "  " + "  ".join([
                f"{'run':>4}", f"{'i_gain':>8}", f"{'f_gain':>8}",
                f"{'Δgain':>8}", f"{'i_rot°':>7}", f"{'f_rot°':>7}",
                f"{'i_tmm':>8}", f"{'f_tmm':>8}", f"{'conv':>5}", f"{'t(s)':>6}",
            ]),
            "  " + "-" * 72,
        ]
        for r in run_table:
            lines.append("  " + "  ".join([
                f"{r['run']:>4}",
                f"{r['init_simlarity']:>8.4f}",
                f"{r['final_gain']:>8.4f}",
                f"{r['gain_delta']:>+8.4f}",
                f"{r['init_rot_err_deg']:>7.2f}",
                f"{r['final_rot_err_deg']:>7.2f}",
                f"{r['init_trans_err_mm']:>8.1f}",
                f"{r['final_trans_err_mm']:>8.1f}",
                f"{r['converge_iter']:>5}",
                f"{r['run_time_s']:>6.1f}",
            ]))

        path.write_text("\n".join(lines) + "\n")

    def _write_global_txt(
        self,
        path:                  Path,
        global_stats:          dict,
        per_patient_condensed: dict,
        total_runs:            int,
    ):
        S = global_stats
        L = self._fmt_stat_line
        # sr_keys = sorted([k for k in S.keys() if k.startswith("SR")], key=lambda x: (int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else float("inf")))
        sr_keys = sorted([k for k in S.keys() if k.startswith("SR") and "deg" not in k], key=lambda x: float(re.search(r"(\d+\.?\d*)", x).group(1)))
        sr_keys += sorted([k for k in S.keys() if k.startswith("SR") and "deg" in k], key=lambda x: float(re.search(r"(\d+\.?\d*)", x).group(1)))
        sr_lines = [f"  {k:<30s}: {S[k]:>6.2f}%" for k in sr_keys]

        lines = [
            "=" * 72,
            f"  GLOBAL SUMMARY  —  {total_runs} runs across "
            f"{len(per_patient_condensed)} patient(s)",
            "=" * 72,
            "",
            "  GAINS",
            L("  initial gain",      S, "init_simlarity"),
            L("  final gain",        S, "final_gain"),
            L("  gain delta",        S, "gain_delta"),
            f"  Runs where gain improved   : {S['pct_gain_improved']}%",
            "",
            "  ROTATION ERROR",
            L("  initial",           S, "init_rot_err_deg",    "°"),
            L("  final",             S, "final_rot_err_deg",   "°"),
            L("  improvement",       S, "rot_improvement_deg", "°"),
            f"  Runs where rot improved    : {S['pct_rot_improved']}%",
            "",
            "  TRANSLATION ERROR",
            L("  initial",           S, "init_trans_err_mm",    "mm"),
            L("  final",             S, "final_trans_err_mm",   "mm"),
            L("  improvement",       S, "trans_improvement_mm", "mm"),
            f"  Runs where trans improved  : {S['pct_trans_improved']}%",
            "",
            "  SUCCESS RATES",
            *sr_lines,
            "",
            "  OPTIMISER",
            L("  converge iter",     S, "converge_iter"),
            L("  run time",          S, "run_time_s", "s"),
            "",
            "  PER-PATIENT BREAKDOWN",
            "  " + f"{'patient':<16}" + "  ".join([
                f"{'f_gain':>8}", f"{'f_rot°':>8}", f"{'f_tmm':>8}",
                f"{'%gain↑':>7}", f"{'%rot↑':>7}", f"{'%trn↑':>7}",
            ]),
            "  " + "-" * 70,
        ]
        for pat, s in per_patient_condensed.items():
            lines.append(
                f"  {pat:<16}" + "  ".join([
                    f"{s['final_gain__mean']:>8.4f}",
                    f"{s['final_rot_err_deg__mean']:>8.2f}",
                    f"{s['final_trans_err_mm__mean']:>8.1f}",
                    f"{s['pct_gain_improved']:>7.1f}%",
                    f"{s['pct_rot_improved']:>7.1f}%",
                    f"{s['pct_trans_improved']:>7.1f}%",
                ])
            )

        path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    # Ensure reproducible pseudo-random behavior optionally
    # torch.manual_seed(42)
    # np.random.seed(42)

    args = parse_args()
    pipeline = PipelineExperiment(args)
    pipeline.run(args.exp_type)
