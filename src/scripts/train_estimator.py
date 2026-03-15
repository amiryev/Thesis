import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

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

from src.config import default as config
from src.models.encoder import XrayEncoder
from src.models.estimator import PositionEstimator
from src.data.dataset import PoseDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices
from src.utils.loss import PositionLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train PositionEstimator (Pose Regression)")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Dataset / Patient
    parser.add_argument("--patient-id", type=str, default="4", help="Patient ID for estimating pose")
    parser.add_argument("--steps", type=int, default=5, help="Grid steps for valid pose generation")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Root directory containing CT/ and CRM/")
    
    # Pretrained Encoder
    parser.add_argument("--encoder-ckpt", type=Path, help="Path to trained encoder checkpoint (required)")
    parser.add_argument("--patch-size", type=int, default=config.PATCH_SIZE, help="Patch size used in encoder")

    # Paths
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/estimator"), help="Directory to save checkpoints/logs")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")

    return parser.parse_args()


def load_encoder(encoder, ckpt_path, rank, logger):
    """Safely loads encoder weights handling DDP prefixes."""
    if not ckpt_path or not ckpt_path.exists():
        if rank == 0: logger.warning(f"Encoder checkpoint {ckpt_path} not found! Using random init (Not Recommended).")
        return encoder

    if rank == 0: logger.info(f"Loading encoder from {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=encoder.device, weights_only=True)
    
    # Extract state dict
    sd = ckpt['model_sd'] if 'model_sd' in ckpt else ckpt
    
    # Clean 'module.' prefix if present
    new_sd = OrderedDict()
    for k, v in sd.items():
        name = k.replace('module.', '')
        new_sd[name] = v
        
    encoder.load_state_dict(new_sd, strict=True)
    return encoder


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, args, rank, logger):
    model.train()
    meters = AverageMeter()
    
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    else:
        pbar = loader

    for i, (rot_gt, trans_gt) in enumerate(pbar):
        rot_gt = rot_gt.to(device, non_blocking=True)
        trans_gt = trans_gt.to(device, non_blocking=True)
        vec_gt = torch.cat([rot_gt, trans_gt], 1)

        # 1. Generate Ground Truth Projection
        # This uses the DiffDRR functionality inside PositionEstimator
        # The model is wrapped in DDP, so we access underlying module attributes
        # Be careful: accessing .module is necessary to call methods not in forward() 
        with torch.no_grad():
            proj_gt = model.module.project(vec_gt) 
        
        optimizer.zero_grad()
        
        # 2. Predict Pose from Projection
        # The forward pass of PositionEstimator takes a projection and estimates pose
        # (It uses the internal self.encoder to get features first)
        proj_pred, vec_pred = model(proj_gt)
        
        # 3. Compute Loss (Pose Error)
        loss = criterion(vec_pred, vec_gt)
        
        loss.backward()
        optimizer.step()
        
        meters.update(loss.item(), rot_gt.size(0))
        
        if rank == 0:
            pbar.set_postfix(loss=f"{meters.avg:.5f}")
            
    return meters.avg


def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    # --- 1. Setup ---
    DDPHelper.setup(rank, world_size)
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", rank)
    
    logger = None
    if rank == 0:
        output_subt = args.output_dir / args.patient_id
        output_subt.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = setup_logger("train_estimator", output_subt / f"train_{timestamp}.log")
        logger.info(f"Starting Estimator Training for Patient {args.patient_id}")

    # --- 2. Model Initialization ---
    
    # A. Encoder
    # We must instantiate the encoder structure first to pass it to the estimator
    encoder = XrayEncoder(
        device=device, 
        size=config.IMAGE_SIZE, 
        patch_size=args.patch_size
    ).to(device)
    
    # Load pretrained weights
    encoder = load_encoder(encoder, args.encoder_ckpt, rank, logger)

    # B. Position Estimator
    ct_file = args.data_dir / "CT" / f"{args.patient_id}.nii.gz"
    crm_file = args.data_dir / "CRM" / f"{args.patient_id}.png"
    
    if not ct_file.exists() and rank == 0:
        logger.error(f"CT file missing: {ct_file}")
    
    model = PositionEstimator(
        encoder=encoder,
        dicom_file=str(ct_file),
        crm_path=str(crm_file),
    ).to(device)
    
    # Wrap DDP
    # We use find_unused_parameters=True if parts of the encoder (like the decoder) aren't used for regression
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # --- 3. Optimizer & Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = PositionLoss().to(device)

    # --- 4. Dataset ---
    # The PoseDataset generates random valid poses for the specific patient loaded in the model
    # We access `model.module.ct` because `model` is the DDP wrapper
    dataset = PoseDataset(
        model.module.ct, 
        device="cpu", # Generate grid on CPU to save GPU memory
        steps=args.steps, 
        min_intersections=500
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=0 # Robustness
    )

    # --- 5. Resume ---
    start_epoch = 0
    best_loss = float('inf')
    ckpt_manager = CheckpointManager(args.output_dir / args.patient_id, rank)
    
    if args.resume:
        ckpt = ckpt_manager.load(args.resume, device)
        if ckpt:
            model.load_state_dict(ckpt['model_sd'])
            optimizer.load_state_dict(ckpt['optim_sd'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            if rank == 0: logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.5f}")

    # --- 6. Training Loop ---
    if rank == 0: logger.info("Training loop started...")
    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch, args, rank, logger
        )
        
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
            
            # Checkpointing
            state = {
                'epoch': epoch,
                'model_sd': model.state_dict(),
                'optim_sd': optimizer.state_dict(),
                'best_loss': best_loss,
                'patient_id': args.patient_id
            }
            
            ckpt_manager.save(f"estimator_last", state, is_best=False)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                state['best_loss'] = best_loss
                ckpt_manager.save(f"estimator_best", state, is_best=True)
                logger.info(f"New best model saved.")

    if rank == 0: logger.info("Training complete.")
    DDPHelper.cleanup()


def main():
    args = parse_args()
    
    # Set visible devices BEFORE any torch.cuda calls or spawning
    set_visible_devices(args.gpus)
    
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Launching DDP Estimator Training on {world_size} GPUs.")
        DDPHelper.spawn(train_worker, world_size, args=(args,))
    else:
        print("No CUDA device found. Running on CPU.")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
