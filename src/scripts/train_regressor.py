import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import numpy as np

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_euler_angles

from src.core.pose_regressor import PoseRegressor
from src.data.dataset import DRRMetadataDataset, RepeatDataset
from src.data.generate_dataset_gpt import load_volume, normalize_and_save
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices
from src.utils.image_processing import RandomGamma, RandomGaussianNoise, RandomGaussianBlur, RandomContrast, RandomSpatialJitter
from src.utils.loss import poseConsistencyLoss, compute_geodesic_distance
import src.utils.config as config

def parse_args():
    """
    Parses command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser("Train Supervised Pose Regressor")
    parser.add_argument("--data_dir",    type=str, default=Path(config.DATA_DIR),   help="Directory encompassing patient folders with meta files.")
    parser.add_argument("--output_dir",  type=str, default=Path(config.OUTPUT_DIR), help="Root dir logic for outputs.")
    parser.add_argument("--ckpt_dir",    type=str, default=Path(config.CKPT_DIR),   help="Path retaining epoch weight sets.")
    parser.add_argument("--test_ids",    type=str, default=None,                    help="Patient ids to test, if None use all.")
    parser.add_argument("--num_workers", type=int, default=0,                       help="Number of dataloader workers.")

    parser.add_argument("--batch_size",    type=int,   default=32,   help="Per-GPU batch size.")
    parser.add_argument("--lr",            type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight_decay",  type=float, default=1e-3, help="Weight decay for AdamW.")
    parser.add_argument("--epochs",        type=int,   default=100,  help="Total training epochs.")
    parser.add_argument("--trans_weight",  type=float, default=0.5,  help="Lambda for translation loss.")
    parser.add_argument("--const_weight",  type=float, default=0.3,  help="Lambda for translation loss.")
    parser.add_argument("--repeats",       type=int,   default=0,    help="Augmentation repeats per sample (0 = disabled).")

    parser.add_argument("--ddp",    action="store_true",               help="Use DistributedDataParallel across all visible GPUs.")
    parser.add_argument("--test",   action="store_true",               help="Run standalone evaluation instead of training.")
    parser.add_argument("--resume", action="store_true",               help="Continue sequence from exact path.")
    parser.add_argument("--gpus",   type=str,            default=None, help="Comma-separated list of GPU IDs to use (e.g. '0,1,3').")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Handles supervised training loops predicting 6D rotation and 3D translation.

    Supports both single-GPU (rank=0, world_size=1) and multi-GPU DDP execution.
    In DDP mode each process creates its own Trainer instance with the appropriate
    rank so that samplers, devices, and logging are all process-local.
    """

    def __init__(self, args, rank: int = 0, world_size: int = 1):
        self.args = args

        # State tracking
        self.start_epoch = 0
        self.best_loss   = float("inf")
        self.history     = {}

        # DDP parameters
        self.rank       = rank
        self.world_size = world_size
        self.is_main    = (rank == 0)

        # Device: each DDP rank owns one GPU; single-GPU falls back to config.DEVICE
        if args.ddp:
            self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            self.device = config.DEVICE

        # Logger setup: only rank 0 writes to console/file
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.datetime.now().strftime('%d_%m_%H_%M')
        self.output_dir = (Path(getattr(config, "OUTPUT_DIR", args.output_dir)) / f"test_results_{timestamp}")
        if self.args.test == False:
            log_file  = Path(args.ckpt_dir) / f"Regressor_{timestamp}.log" if self.is_main else None
            self.logger = setup_logger("Train Regressor", log_file, rank=self.rank)
            self.logger.info("Starting Pose Regressor Training")
        else:
            os.makedirs(self.output_dir,  exist_ok=True)
            log_file  = self.output_dir / f"Regressor_test.log" if self.is_main else None
            self.logger = setup_logger("Test Regressor", log_file, rank=self.rank)
            self.logger.info("Starting Pose Regressor Testing")

        self.logger.info(f"Arguments: {vars(args)}")
        
        # Setup data loaders and set splits
        self.setup_dataloaders()

        # Setup model and optimizer
        self.setup_model()

        self.load_checkpoint()

    def setup_model(self):
        """
        Initialises the model, optimizer, and learning-rate scheduler.

        Note: DDP wrapping is intentionally done here (before checkpoint loading
        in load_checkpoint) so state-dict keys remain compatible when reloading.
        """
        # Per-axis translation loss weights: y-axis (superior/inferior) is
        self.trans_weight = self.args.trans_weight * torch.tensor([1.0, 0.2, 1.0], device=self.device)

        # Approximate max displacements per axis (mm) used to normalise
        # self.trans_scale = torch.tensor([40.0, 50.0, 40.0], device=self.device)
        self.trans_scale = 50.0
        self.const_weight = self.args.const_weight
        
        # Model & optimiser
        self.model = PoseRegressor(dropout=0.3).to(self.device)

        # Wrap model with DDP after checkpoint is loaded into the bare module
        if self.args.ddp:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)
               
        # Optimizer and scheduler are only needed during training, not evaluation
        if self.args.test is False:
            # Helper to handle both DDP and standard models
            model_to_access = self.model.module if hasattr(self.model, 'module') else self.model
            # Split parameters into two groups so we can apply different regularisation:
            backbone_params = list(model_to_access.backbone.parameters()) + \
                            list(model_to_access.sobel.parameters())
                            # list(model_to_access.positional_encoding) +\
                            # list(model_to_access.mask_token)
                            # list(self.model.feature_dropout.parameters()) + \
                            # list(self.model.pooling.parameters())
                            
            head_params     = list(model_to_access.mlp.parameters()) + \
                            list(model_to_access.rotation.parameters()) + \
                            list(model_to_access.translation.parameters())            
            self.optimizer = torch.optim.AdamW([
                {"params": backbone_params, "weight_decay": self.args.weight_decay},          # keep current
                {"params": head_params,     "weight_decay": self.args.weight_decay * 10},     # 10× stronger on MLP
            ], lr=self.args.lr)

            # OneCycleLR: separate max_lr per param group so the backbone is driven at 10 % of the head LR (fine-tuning regime).
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[self.args.lr * 0.1, self.args.lr],
                epochs=self.args.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,        # 10% warmup
                anneal_strategy='cos',
            )

    def load_checkpoint(self):
        """
        Loads a previously saved checkpoint and restores model/optimizer state.

        Checkpoint resolution order (first match wins):
          1. ``regressor_patient<id>.pth``   patient-specific fine-tune checkpoint
          2. ``regressor_best.pth``          best validation-loss checkpoint
          3. ``regressor_last.pth``          most recent epoch checkpoint

        When resuming training the optimizer state, starting epoch, best loss,
        and loss history are all restored so training can continue seamlessly.
        In test-only mode only the model weights are loaded.
        """
        # Create checkpoints directory
        self.ckpt_dir = Path(self.args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint manager (rank-0 only writes)
        self.ckpt_mgr = CheckpointManager(self.ckpt_dir, rank=self.rank)
        
        # Load checkpoint BEFORE DDP wrapping so state_dict keys stay compatible
        if self.args.resume or self.args.test:
            # Prefer best model, fall back to last, then allow explicit --resume override
            ckpt_path = Path(self.ckpt_dir) / f"regressor_patient{self.test_ids[0]}.pth" if self.args.test_ids is not None else Path(self.ckpt_dir) / "regressor_best.pth"
            if not ckpt_path.exists():
                ckpt_path = Path(self.ckpt_dir) / "regressor_best.pth"
                if not ckpt_path.exists():
                    ckpt_path = Path(self.ckpt_dir) / "regressor_last.pth"

            ckpt = self.ckpt_mgr.load(ckpt_path, self.device)
            if ckpt is None:
                self.logger.warning(f"Checkpoint {ckpt_path} not found, starting from scratch.")
                return

            # Log checkpoint loading message
            self.logger.info(f"Loading previous state from: {ckpt_path}")
            self.model.load_state_dict(ckpt["model"])

            if ("optimizer" in ckpt) and (self.args.test == False):
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.start_epoch = ckpt["epoch"] + 1
                self.args.epochs = self.start_epoch + self.args.epochs
                if "best_loss" in ckpt:
                    self.best_loss = ckpt["best_loss"]
                if "history" in ckpt:
                    self.history = ckpt["history"]
                self.logger.info(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.5f}")

    def split_dataset(self, train_p=0.8, val_p=0.2):
        """
        Deterministically splits available patient IDs into train and validation sets.

        Args:
            train_p (float): Fraction of patients assigned to the training set.
            val_p   (float): Fraction of patients assigned to the validation set.

        Returns:
            tuple[list[int], list[int]]: (train_ids, val_ids) — shuffled patient ID lists.
        """
        # Collect all patient IDs from the data directory folder names (patient_<id>)
        ids = sorted(int(name.split("_")[1]) for name in os.listdir(self.args.data_dir) if name.startswith("patient_"))
        num_ids = len(ids)

        # Fixed seed for reproducibility across runs and ranks
        torch.manual_seed(42)
        perm = torch.randperm(num_ids).tolist()
        ids = [ids[i] for i in perm]

        num_train_ids = int(num_ids * train_p)

        return ids[:num_train_ids], ids[num_train_ids:]

    def load_dataset(self, data_dir, ids, batch_size, transform=None, shuffle=False):
        """
        Constructs a DataLoader for the given patient IDs.

        In DDP mode a ``DistributedSampler`` is used so each rank receives a
        non-overlapping shard of the dataset.  When DDP is disabled the sampler
        is ``None`` and ``shuffle`` is forwarded directly to the DataLoader.

        Args:
            data_dir  (str | Path): Root directory containing patient sub-folders.
            ids       (list[int]):  Patient IDs to include in this dataset split.
            batch_size (int):       Number of samples per batch.
            transform (callable):  Optional torchvision transform applied to images.
            shuffle   (bool):      Whether to shuffle samples each epoch.

        Returns:
            torch.utils.data.DataLoader: Ready-to-iterate DataLoader.
        """
        dataset = DRRMetadataDataset(
            root_dir=data_dir, 
            transform=transform, 
            return_pose=True, 
            valid_ids=ids)
        
        # DistributedSampler partitions the dataset across ranks; when not using
        # DDP we fall back to None so the DataLoader uses its built-in shuffling.
        data_sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
        ) if self.args.ddp else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=data_sampler,
            num_workers=self.args.num_workers,
            persistent_workers=(self.args.num_workers>0),
            pin_memory=True,
            shuffle=(shuffle and (data_sampler is None)),
        )
        return dataloader

    def setup_dataloaders(self):
        """
        Prepares train/val (or test) DataLoaders and sets ``self.test_ids``.

        Behaviour depends on the run mode:
        - **Training**: splits all patients into train/val sets, builds both loaders.
        - **Test**: builds a single test loader for the patients given by ``--test_ids``
          (or all available patients if ``--test_ids`` is not provided).

        When ``--repeats > 1`` (consistency loss enabled) the effective batch size
        fed to the DataLoader is divided by ``repeats`` so that after in-batch
        augmentation expansion the GPU memory footprint stays the same.
        """
        # Note: augmentations are defined here so that they are shared across
        # all augmented repetitions of the same image within a batch.
        self.logger.info(f"Preparing Datasets from: {self.args.data_dir}")

        # Dataset parameters
        self.repeats = self.args.repeats
        self.transforms = T.Compose([
            RandomGamma(range=(0.5, 2.0)),                 # simulates different dose/kV settings
            RandomGaussianNoise(std_range=(0, 0.05)),      # simulates quantum noise
            RandomGaussianBlur(sigma_range=(0.0001, 1.5)), # simulates MTF differences
            RandomContrast(range=(0.7, 1.3)),              # simulates detector response variation
            RandomSpatialJitter(),
        ]) 

        # Split patient ids to train\val\test sets
        if self.args.resume and self.args.test_ids:
            ids_val = [6749823] #None
            ids_train = [int(x) for x in self.args.test_ids.split(',')]
        else:
            ids_train, ids_val = self.split_dataset(train_p=0.9, val_p=0.1)
            # ids_train, ids_val = [7,9,10,11,13], [8,14]
        self.logger.info(f"Using patient IDs: {ids_train}\{ids_val}")
        
        # If consistency loss is used divide batch_size by repeats
        batch_size = (self.args.batch_size // self.repeats) if self.repeats > 1 else self.args.batch_size

        self.test_ids = None if self.args.test_ids is None else [int(x) for x in self.args.test_ids.split(',')]
        if self.args.test == False:
            self.train_loader = self.load_dataset(
                data_dir=self.args.data_dir, 
                ids=ids_train, 
                batch_size=batch_size, 
                transform=None,
                shuffle=True,
            )
            self.val_loader = self.load_dataset(
                data_dir=self.args.data_dir, 
                ids=ids_val, 
                batch_size=self.args.batch_size, 
                transform=None,
            )
        else:
            print_ids = self.test_ids if self.args.test_ids is not None else "All"
            self.logger.info(f"Testing data using patients IDs: {print_ids}")
            self.test_loader = self.load_dataset(
                data_dir=self.args.data_dir, 
                ids=self.test_ids, 
                batch_size=self.args.batch_size, 
                transform=None,

            )

    def plot_loss_curve(self, save_path: Path):
        """
        Plots training loss curves and saves to file. Only called on rank 0.
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history["train_loss"]) + 1)

        plt.plot(epochs, self.history["train_loss"],       label="Total Loss",        marker="o",  color="blue")
        plt.plot(epochs, self.history["train_rot_loss"],   label="Rotational Loss",   marker="x",  linestyle="--", color="orange")
        plt.plot(epochs, self.history["train_trans_loss"], label="Translational Loss",marker="x",  linestyle="--", color="green")
        plt.plot(epochs, self.history["train_const_loss"], label="Consistency Loss",  marker="x",  linestyle="--", color="purple")
        plt.plot(epochs, self.history["val_loss"],         label="Validation Loss",   marker="o",  color="red")

        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Pose Regressor Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def compute_loss(self, rot_6d_pred, trans_pred, poses_gt, is_train=True):
        """
        Computes the combined pose regression loss.

        The total loss is a weighted sum of three terms:
          - **Rotation loss**: mean geodesic (angular) distance between the
            predicted and ground-truth rotation matrices (in radians).
          - **Translation loss**: per-axis Smooth-L1 loss, normalised by
            ``self.trans_scale`` and weighted per-axis by ``self.trans_weight``.
          - **Consistency loss** (optional): penalises disagreement between the
            multiple augmented views of the same image within a batch.  
            The weight is linearly ramped from 0 to ``const_weight`` over the first 10 epochs.

        Args:
            rot_6d_pred (Tensor): Predicted 6D rotation representation, shape (B, 6).
            trans_pred  (Tensor): Predicted 3D translation in mm, shape (B, 3).
            poses_gt    (Tensor): Ground-truth pose [euler_ZXY | translation], shape (B, 6).
            is_train    (bool):   If False, consistency loss is skipped and
                                  per-sample rotation errors (not the mean) are
                                  returned for downstream success-rate computation.

        Returns:
            tuple:
                - loss            (Tensor): Scalar total loss.
                - loss_rot        (Tensor): Mean rotation loss (scalar) during training;
                                           per-sample errors (B,) during evaluation.
                - loss_trans      (Tensor): Mean (normalised) translation loss (scalar).
                - consistency_loss(Tensor): Consistency loss scalar (0 when disabled).
        """
        euler_gt = poses_gt[:, :3]   # ZXY Euler angles (radians)
        trans_gt = poses_gt[:, 3:]   # translation in mm

        # Convert predictions and GT to rotation matrices for geodesic distance
        rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
        rot_matrix_gt   = euler_angles_to_matrix(euler_gt, convention="ZXY")

        # Geodesic distance returns per-sample angular errors (radians)
        loss_rot = compute_geodesic_distance(rot_matrix_pred, rot_matrix_gt)
        loss_rot_mean = loss_rot.mean()

        # Smooth-L1 computed per axis then normalised; reduction="none" retains
        # the per-axis shape (B, 3) so per-axis weights can be applied.
        loss_trans_by_axis = F.smooth_l1_loss(trans_pred, trans_gt, reduction="none") / self.trans_scale
        loss_trans         = loss_trans_by_axis.mean()

        if (self.repeats > 1) and (is_train == True):
            # Linearly ramp the consistency weight from 0 → const_weight over the first 10 epochs
            const_weight = min(self.const_weight, (self.epoch / 10) * self.const_weight)
            consistency_loss = poseConsistencyLoss(torch.cat([rot_6d_pred, trans_pred], dim=1), repeats=self.repeats)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
            const_weight = 0.0

        loss = loss_rot_mean + (self.trans_weight * loss_trans_by_axis).mean() + (const_weight * consistency_loss)

        # During testing return per-sample rotation errors (not the mean) so the
        # caller can compute per-sample success rates and per-patient statistics.
        if self.args.test:
            return loss, loss_rot, loss_trans_by_axis, consistency_loss    

        return loss, loss_rot_mean, loss_trans, consistency_loss

    def train_epoch(self):
        """
        Runs one full pass over the training DataLoader.

        When ``--repeats > 1`` each original image is duplicated and augmented
        ``repeats - 1`` times in-batch so that the consistency loss can penalise
        prediction disagreement across views.  The augmented copies are produced
        on-the-fly on the GPU using the transforms defined in ``setup_dataloaders``.

        Returns:
            tuple[float, float, float, float]:
                (avg_total_loss, avg_rot_loss, avg_trans_loss, avg_consistency_loss)
                — batch-size-weighted averages over the entire epoch.
        """
        # Note: only rank 0 renders the tqdm progress bar to avoid duplicated output in DDP mode
        self.model.train()

        loss_meter       = AverageMeter()
        rot_loss_meter   = AverageMeter()
        trans_loss_meter = AverageMeter()
        const_loss_meter = AverageMeter()

        # Only rank 0 shows the progress bar to avoid duplicated output
        pbar = tqdm(self.train_loader, desc="Minibatch Progression", leave=False, disable=not self.is_main)
        for images, poses_gt, _ in pbar:
            images, poses_gt = images.to(self.device), poses_gt.to(self.device)
            B = images.shape[0]

            # if self.repeats > 1:
            #     images_repeated = images.view(-1, 1, images.shape[3], images.shape[4]).to(self.device)
            #     poses_gt = poses_gt.view(-1, poses_gt.shape[2]).to(self.device)
            # else:
            #     images, poses_gt = images.to(self.device), poses_gt.to(self.device)

            if self.repeats > 1:
                # Expand ground-truth labels to match the augmented batch
                poses_gt = poses_gt.repeat_interleave(self.repeats, dim=0)  # (B*R, 6)

                # Create (repeats-1) augmented copies of every image in the batch,
                images_augmented = images.repeat_interleave(self.repeats - 1, dim=0)  # (B*(R-1), C, H, W)
                images_augmented = self.transforms(images_augmented)
                images_augmented = images_augmented.view(B, self.repeats - 1, *images.shape[1:])

                # Concatenate the original image and augmented images so the final tensor has shape (B*repeats, C, H, W).
                images_repeated = torch.cat([images.unsqueeze(1), images_augmented], dim=1).reshape(B * self.repeats, *images.shape[1:])
            else:
                images_repeated = images

            self.optimizer.zero_grad()
            rot_6d_pred, trans_pred = self.model(images_repeated)

            loss, loss_rot, loss_trans, consistency_loss = self.compute_loss(rot_6d_pred, trans_pred, poses_gt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            loss_meter.update(loss.item(), B)
            rot_loss_meter.update(loss_rot.item(), B)
            trans_loss_meter.update(loss_trans.item(), B)
            const_loss_meter.update(consistency_loss.item(), B)

            pbar.set_postfix({
                "Loss":  f"{loss.item():.4f}",
                "Rot":   f"{loss_rot.item():.4f}",
                "Trans": f"{loss_trans.item():.4f}",
                "Const": f"{consistency_loss.item():.4f}",
            })

        return loss_meter.avg, rot_loss_meter.avg, trans_loss_meter.avg, const_loss_meter.avg

    @torch.no_grad()
    def validate(self):
        """
        Evaluates the model on the validation split (no gradient computation).

        Returns:
            tuple[float, float, float]:
                (avg_total_loss, avg_rot_loss, avg_trans_loss)
                — batch-size-weighted averages over the entire validation set.
        """
        self.model.eval()

        loss_meter       = AverageMeter()
        rot_loss_meter   = AverageMeter()
        trans_loss_meter = AverageMeter()
        pbar = tqdm(self.val_loader, desc="Validation Progression", leave=False, disable=not self.is_main)
        for images, poses_gt, _ in pbar:
            images, poses_gt = images.to(self.device), poses_gt.to(self.device)
            B = images.size(0)

            rot_6d_pred, trans_pred = self.model(images)

            loss, loss_rot, loss_trans, _ = self.compute_loss(rot_6d_pred, trans_pred, poses_gt, is_train=False)

            loss_meter.update(loss.item(), B)
            rot_loss_meter.update(loss_rot.item(), B)
            trans_loss_meter.update(loss_trans.item(), B)

        return loss_meter.avg, rot_loss_meter.avg, trans_loss_meter.avg

    def run_train(self):
        """
        Main training loop: iterates over epochs, records metrics, and saves checkpoints.
        """
        self.logger.info("Training sequence started.")

        # Loss values
        best_trans_loss = torch.inf
        best_rot_loss = torch.inf

        for epoch in range(self.start_epoch, self.args.epochs):
            self.logger.info(f"--- Epoch [ {epoch+1} / {self.args.epochs} ] Started ---")
            self.epoch = epoch

            # Ensure each rank sees a different shuffle each epoch (DDP requirement)
            if self.args.ddp:
                self.train_loader.sampler.set_epoch(epoch)
                self.val_loader.sampler.set_epoch(epoch)

            train_loss, train_rot_loss, train_trans_loss, train_const_loss = self.train_epoch()
            self.logger.info(
                f"[Train] Total: {train_loss:.4f} | Rot: {train_rot_loss:.4f} | "
                f"Trans: {train_trans_loss:.4f} | Const: {train_const_loss:.4f}"
            )

            val_loss, val_rot_loss, val_trans_loss = self.validate()
            self.logger.info(
                f"[Val]   Total: {val_loss:.4f} | Rot: {val_rot_loss:.4f} | Trans: {val_trans_loss:.4f}"
            )

            # Store history
            self.history.setdefault("train_loss",       []).append(train_loss)
            self.history.setdefault("train_rot_loss",   []).append(train_rot_loss)
            self.history.setdefault("train_trans_loss", []).append(train_trans_loss)
            self.history.setdefault("train_const_loss", []).append(train_const_loss)
            self.history.setdefault("val_loss",         []).append(val_loss)
            self.history.setdefault("val_rot_loss",     []).append(val_rot_loss)
            self.history.setdefault("val_trans_loss",   []).append(val_trans_loss)

            # Plot (rank 0 only)
            if self.is_main:
                self.plot_loss_curve(self.ckpt_dir / "regressor_loss_curve.png")

            # Unwrap DDP module for a device-agnostic state dict
            model_sd = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            pid = self.test_ids[0] if self.args.resume is True else None
            state = {
                "model":     model_sd,
                "epoch":     epoch,
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "history":   self.history,
                "patient_id": pid,
            }

            # Save latest checkpoint (atomic, rank 0 only)
            path = self.ckpt_mgr.save("regressor_last", state, pid)
            self.logger.info(f'Saved latest model → {path}')

            # Save best checkpoint separately (Option A: explicit second save)
            if val_loss < self.best_loss:
                self.best_loss       = val_loss
                state["best_loss"]   = self.best_loss
                self.ckpt_mgr.save("regressor_best", state)
                self.logger.info(f"*** New best model saved with VAL Loss {self.best_loss:.4f} ***")
                self.logger.info(f"*************************************************")

            # Save best rotation model
            if val_rot_loss < best_rot_loss:
                best_rot_loss = val_rot_loss
                state["best_loss"] = val_loss
                self.ckpt_mgr.save("regressor_rot", state)
                self.logger.info(f"*** New best rotation model saved with VAL Loss {val_rot_loss:.4f} ***")
                self.logger.info(f"*************************************************")
            
            # Save best translation model
            if val_trans_loss < best_trans_loss:
                best_trans_loss = val_trans_loss
                state["best_loss"] = val_loss
                self.ckpt_mgr.save("regressor_trans", state)
                self.logger.info(f"*** New best translation model saved with VAL Loss {val_trans_loss:.4f} ***")
                self.logger.info(f"*************************************************")

        self.logger.info("Training sequence completed.")

    def load_volume(self, patient_id, rot, trans):
        master_path = os.path.join(self.args.data_dir, 'master_index.json')
        
        with open(master_path, 'r') as f:
            patients = json.load(f)
        ct_path    = [p['ct_path'] for p in patients if (p['id'] == patient_id)]
        ct_path = ct_path[0]
        vol = load_volume(ct_path)

        import nibabel as nib
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        tmp_path = Path(tmp.name)

        nib.save(vol, str(tmp_path))
        subject    = read(volume=tmp_path, orientation="AP", center_volume=True)
        tmp_path.unlink(missing_ok=True)

        # normalize_and_save
        # subject    = read(volume=str(ct_path), orientation="AP", center_volume=True)
        drr        = DRR(subject, sdd=config.SDD, height=config.IMAGE_SIZE, delx=config.DELX)
        projection = drr(rot.to("cpu"), trans.to("cpu"), parameterization="euler_angles", convention="ZXY")
        # projection = drr(rot_6d_pred, trans_pred, parameterization="rotation_6d")

        out = normalize_and_save(projection, path=None, save=False)
        out = 1.0 - out.detach().cpu().squeeze().numpy()
        return out

    def save_visualization(self, patient_id, loss, original, image_path, rot_gt, rot_pred, trans_gt, trans_pred, rot_6d_pred=None):
        """
        Renders and saves a side-by-side comparison of the GT and predicted DRR projections.

        The predicted pose is used to re-render the CT scan with DiffDRR so the
        visual quality of the prediction can be assessed qualitatively.  The
        output figure includes annotated Euler angles and translation vectors for
        both GT and prediction.

        Args:
            patient_id  (Tensor | int): Scalar patient ID used for the output sub-folder.
            loss        (float):        Scalar loss value shown in the figure title.
            original    (Tensor):       Original DRR image tensor (C, H, W) or (1, H, W).
            image_path  (str):          Full path to the source DRR file; the serial
                                        number is parsed from the filename to name the
                                        output file consistently.
            rot_gt      (np.ndarray):   Ground-truth Euler angles in radians, shape (3,).
            rot_pred    (any):          Unused; kept for API compatibility.
            trans_gt    (np.ndarray):   Ground-truth translation in mm, shape (3,).
            trans_pred  (Tensor):       Predicted translation in mm, shape (3,).
            rot_6d_pred (Tensor):       Predicted 6D rotation, shape (6,).  Converted
                                        internally to Euler angles for DiffDRR rendering.
        """
        save_dir = self.output_dir / f"patient_{patient_id.item()}"
        save_dir.mkdir(parents=True, exist_ok=True)

        rot_matrix_pred = rotation_6d_to_matrix(rot_6d_pred)
        rot_pred  = matrix_to_euler_angles(rot_matrix_pred, convention="ZXY")

        img = original.detach().cpu().squeeze().numpy()
        out = self.load_volume(patient_id, rot_pred, trans_pred)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap="gray"); ax[0].axis("off"); ax[0].set_title("GT Pose Projection")
        ax[1].imshow(out, cmap="gray"); ax[1].axis("off"); ax[1].set_title("Predicted Pose Projection")

        pred_angles = np.rad2deg(rot_pred.cpu().numpy())
        gt_angles   = np.rad2deg(rot_gt)
        info_text   = (
            f"Loss: {loss:.4f}\n"
            f"GT Rot (deg): [{gt_angles[0, 0]:.1f}, {gt_angles[0, 1]:.1f}, {gt_angles[0, 2]:.1f}] | "
            f"Trans (mm): [{trans_gt[0, 0]:.1f}, {trans_gt[0, 1]:.1f}, {trans_gt[0, 2]:.1f}]\n"
            f"PR Rot (deg): [{pred_angles[0, 0]:.1f}, {pred_angles[0, 1]:.1f}, {pred_angles[0, 2]:.1f}] | "
            f"Trans (mm): [{trans_pred[0, 0]:.1f}, {trans_pred[0, 1]:.1f}, {trans_pred[0, 2]:.1f}]"
        )
        fig.suptitle(info_text, fontsize=10)

        serial    = int(image_path.split("drr_")[-1].split(".png")[0])
        save_file = save_dir / f"pose_eval_{serial:03d}.png"
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()

    @torch.no_grad()
    def evaluate(self, success_rot_deg=5.0, success_trans_mm=10.0, num_vis_per_batch=5):
        """
        Runs inference on the test set and computes pose estimation success metrics.

        A prediction is considered a *success* if both the rotation error is below
        ``success_rot_deg`` degrees AND the translation error is below
        ``success_trans_mm`` mm.  Two variants of the translation criterion are
        reported: full 3-D distance and the XZ-plane distance (omitting the
        superior/inferior Y axis which is less clinically relevant).

        Per-patient error breakdowns are printed to stdout.

        Args:
            success_rot_deg   (float): Rotation error threshold for success (degrees).
            success_trans_mm  (float): Translation error threshold for success (mm).
            num_vis_per_batch (int):   Maximum number of samples to visualise per
                                       randomly selected batch.

        Returns:
            dict: {
                "mean_rot_err_deg"         : float,
                "mean_trans_err_mm"        : float,
                "mean_trans_err_xz_mm"     : float,
                "success_rate_<R>deg_<T>mm": float (percentage),
                "xz_success_rate_<R>deg_<T>mm": float (percentage),
            }
        """
        self.model.eval()

        all_rot_errs   = []
        all_trans_errs = []
        xz_trans_errs  = []

        torch.seed()
        visualize_idxs = torch.randint(0, len(self.test_loader), size=(5,))

        pbar = tqdm(self.test_loader, desc="Testing Evaluation", leave=False)
        for batch_idx, (images, poses_gt, samples) in enumerate(pbar):
            images, poses_gt = images.to(self.device), poses_gt.to(self.device)

            rot_6d_pred, trans_pred = self.model(images)

            euler_gt      = poses_gt[:, :3]
            trans_gt      = poses_gt[:, 3:]
            trans_dists    = torch.norm(trans_pred - trans_gt, dim=1)                            # (B)
            trans_dists_xz = torch.norm(trans_pred[:, [0, 2]] - trans_gt[:, [0, 2]], dim=1)      # (B)
            
            loss, rot_dists, loss_trans, _ = self.compute_loss(rot_6d_pred, trans_pred, poses_gt, is_train=False)
            loss = rot_dists + (self.trans_weight * loss_trans).mean(axis=1)

            rot_dists_deg  = torch.rad2deg(rot_dists)

            if batch_idx in visualize_idxs:
                # for v_idx in range(min(num_vis_per_batch, images.size(0))):
                    v_idx = torch.randint(0, images.shape[0], size=(1,))
                    loss_v = loss[v_idx].item()
                    self.save_visualization(
                        patient_id=samples["id"][v_idx],
                        loss=loss_v,
                        original=images[v_idx],
                        image_path=samples["path"][v_idx],
                        rot_gt=euler_gt[v_idx].cpu().numpy(),
                        rot_pred=None,
                        trans_gt=trans_gt[v_idx].cpu().numpy(),
                        trans_pred=trans_pred[v_idx],
                        rot_6d_pred=rot_6d_pred[v_idx],
                    )

            all_rot_errs.append(rot_dists_deg.cpu())
            all_trans_errs.append(trans_dists.cpu())
            xz_trans_errs.append(trans_dists_xz.cpu())

        all_rot_errs   = torch.cat(all_rot_errs)
        all_trans_errs = torch.cat(all_trans_errs)
        xz_trans_errs  = torch.cat(xz_trans_errs)

        mean_rot_err      = all_rot_errs.mean().item()
        mean_trans_err    = all_trans_errs.mean().item()
        mean_xz_trans_err = xz_trans_errs.mean().item()

        # Joint success: both rotation AND full-3D translation must be within thresholds
        success_mask    = (all_rot_errs < success_rot_deg) & (all_trans_errs < success_trans_mm)
        success_rate    = success_mask.float().mean().item() * 100.0

        # XZ success: same rotation criterion but translation measured only in the XZ plane
        xz_success_mask = (all_rot_errs < success_rot_deg) & (xz_trans_errs < success_trans_mm)
        xz_success_rate = xz_success_mask.float().mean().item() * 100.0

        # Per-patient breakdown
        per_patient = {}
        if self.test_ids is None:
            patients_paths = os.listdir(self.args.data_dir)
            self.test_ids = [int(p.split('_')[-1]) for p in patients_paths if os.path.isdir(p)]
        for err_r, err_t, err_xz, pid in zip(all_rot_errs, all_trans_errs, xz_trans_errs, self.test_ids):
            per_patient.setdefault(pid, []).append((err_r, err_t, err_xz))
        for pid, errs in per_patient.items():
            r_errs  = torch.tensor([e[0] for e in errs])
            t_errs  = torch.tensor([e[1] for e in errs])
            xz_errs = torch.tensor([e[2] for e in errs])
            print(f"Patient {pid}: rot={r_errs.mean():.1f}°  trans={t_errs.mean():.1f}mm  xz={xz_errs.mean():.1f}mm")

        return {
            "mean_rot_err_deg":    mean_rot_err,
            "mean_trans_err_mm":   mean_trans_err,
            "mean_trans_err_xz_mm":   mean_xz_trans_err,
            f"success_rate_{success_rot_deg}deg_{success_trans_mm}mm":    success_rate,
            f"xz_success_rate_{success_rot_deg}deg_{success_trans_mm}mm": xz_success_rate,
        }

    def run_test(self, success_rot_deg=5.0, success_trans_mm=10.0, num_vis_per_batch=5):
        """
        Entry point for stand-alone evaluation mode (``--test`` flag).

        Delegates to ``evaluate()`` and logs the aggregated metrics.

        Args:
            success_rot_deg   (float): Rotation success threshold in degrees.
            success_trans_mm  (float): Translation success threshold in mm.
            num_vis_per_batch (int):   Max visualisations saved per selected batch.
        """
        self.logger.info("Standalone Evaluation Sequence Initialization.")
        metrics = self.evaluate(success_rot_deg, success_trans_mm, num_vis_per_batch)
        self.logger.info(f"Test Phase Aggregation Metrics: {metrics}")

# ---------------------------------------------------------------------------
# DDP entry point
# ---------------------------------------------------------------------------

def train_ddp(rank: int, world_size: int, args):
    """
    Per-process entry point for DDP training. Spawned by DDPHelper.spawn().

    Each process:
      1. Initialises the distributed process group and receives its device.
      2. Creates a Trainer bound to its rank/world_size.
      3. Runs the full training loop.
      4. Cleans up the process group.

    Args:
        rank:       Global rank of this process (0 … world_size-1).
        world_size: Total number of GPU processes.
        args:       Parsed argparse.Namespace from the main process.
    """
    try:
        DDPHelper.setup(rank, world_size)
        trainer = Trainer(args, rank=rank, world_size=world_size)
        trainer.run_train()
    finally:
        DDPHelper.cleanup()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Script entry point.  Resolves the run mode and dispatches accordingly:

    - ``--test``      : single-process evaluation using the best saved checkpoint.
    - ``--ddp``       : multi-GPU training via ``torch.multiprocessing.spawn``;
                        requires at least 2 visible CUDA devices.
    - *(default)*     : single-GPU training.
    """
    args = parse_args()

    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.output_dir,  exist_ok=True)

    # Restrict visible GPUs before any CUDA context is created
    set_visible_devices(args.gpus)

    if args.test:
        trainer = Trainer(args)
        trainer.run_test(success_rot_deg=10.0, success_trans_mm=20.0)

    elif args.ddp:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise RuntimeError(
                f"--ddp requires at least 2 visible CUDA devices, but found {world_size}. "
                "Set CUDA_VISIBLE_DEVICES to expose multiple GPUs before launching."
            )
        # Each spawned process calls train_ddp(rank, world_size, args)
        DDPHelper.spawn(train_ddp, world_size, args=(args,))

    else:
        trainer = Trainer(args)
        trainer.run_train()


if __name__ == "__main__":
    main()
