import os
import argparse
from pathlib import Path
import time
import datetime
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.utils import config
from src.core.encoder import XrayEncoder
from src.data.dataset import MultiPatientDRRDataset, DRRMetadataDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices

def parse_args():
    parser = argparse.ArgumentParser(description="Train XrayEncoder (Masked Reconstruction)")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--on_the_fly", type=str, default=False, help="Generate DRR while training or use pre-generated dataset")

    # Encoder Specific
    parser.add_argument("--mask-ratio", type=float, default=0.6, help="Ratio of patches to mask")
    parser.add_argument("--patch-size", type=int, default=config.PATCH_SIZE, help="Size of patches")
    
    # Paths
    parser.add_argument("--ckpt_dir", type=Path, default=Path(config.CKPT_DIR), help="Directory to save checkpoints/logs")
    parser.add_argument("--data_dir", type=Path, default=Path(config.DATA_DIR), help="Root directory containing CT/ and CRM/")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")

    return parser.parse_args()


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes L1 loss only on masked pixels.
    Args:
        pred: (B, 1, H, W) Reconstructed image
        target: (B, 1, H, W) Original image
        pixel_mask: (B, 1, H, W) Binary mask (1 where generated/masked, 0 where kept)
    """
    diff = torch.abs(pred - target) * pixel_mask
    # Normalize by number of masked pixels to avoid scaling issues with mask ratio
    denom = pixel_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def train_one_epoch(model, loader, optimizer, device, epoch, args, rank, logger):
    model.train()
    meters = AverageMeter()
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    else:
        pbar = loader

    for i, batch in enumerate(pbar):
        imgs = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass: 
        # The model internally handles masking and reconstruction
        recon, pixel_mask, _ = model(imgs, mask_ratio=args.mask_ratio)
        
        # Loss calculation
        loss = masked_l1_loss(recon, imgs, pixel_mask)
        
        loss.backward()
        optimizer.step()
        
        meters.update(loss.item(), imgs.size(0))
        
        if rank == 0:
            pbar.set_postfix(loss=f"{meters.avg:.5f}")
            
    return meters.avg


def train_worker(rank: int, world_size: int, args: argparse.Namespace):
    # --- 1. Setup Distributed Environment ---
    DDPHelper.setup(rank, world_size)
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", rank)
    
    # Logging setup (Rank 0 only)
    logger = None
    if rank == 0:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = setup_logger("train_encoder", args.ckpt_dir / f"train_{timestamp}.log")
        logger.info(f"Starting Encoder Training on {world_size} GPUs")
        logger.info(f"Arguments: {vars(args)}")

    # --- 2. Data Loading ---
    data_dir = args.data_dir
    if not data_dir.exists():
        if rank == 0: logger.error(f"CT directory not found at {data_dir}")
        DDPHelper.cleanup()
        return

    if args.on_the_fly is True:
        dataset = MultiPatientDRRDataset(
            data_dir=data_dir,
            device='cpu',
            size=config.IMAGE_SIZE,
            patient_ids=(7,11)
            # Potentially additional params if dataset supports them
        )
    else:
        dataset = DRRMetadataDataset(root_dir=data_dir)

    # Distributed Sampler ensures each GPU sees a different subset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    pw = (args.num_workers > 0)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,     # Increased from 16 to utilize 48GB VRAM
        sampler=sampler, 
        num_workers=args.num_workers,   # Start with 4 workers PER GPU (16 total)
        pin_memory=True,                # Keeps data in "page-locked" memory for faster GPU transfer
        persistent_workers=pw,         # Keeps workers alive between epochs to save time
    )

    # --- 3. Model Setup ---
    model = XrayEncoder(
        device=device, 
        size=config.IMAGE_SIZE, 
        patch_size=args.patch_size
    ).to(device)
    
    # Wrap with DDP
    # find_unused_parameters=False is efficient when all parameters are used in forward pass
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # --- 4. Resume Checkpoint ---
    start_epoch = 0
    best_loss = float('inf')
    ckpt_manager = CheckpointManager(args.ckpt_dir, rank)
    
    if args.resume:
        ckpt = ckpt_manager.load(args.resume, device)
        if ckpt:
            model.load_state_dict(ckpt['model_sd'])
            optimizer.load_state_dict(ckpt['optim_sd'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            if rank == 0: logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.5f}")
        else:
            if rank == 0: logger.warning(f"Checkpoint {args.resume} not found, starting from scratch.")

    # --- 5. Training Loop ---
    if rank == 0: logger.info("Training loop started...")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch) # Crucial for shuffling in DDP
        
        avg_loss = train_one_epoch(
            model, loader, optimizer, device, epoch, args, rank, logger
        )
        
        if rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Elapsed: {elapsed:.1f}s")
            
            # Save Checkpoints
            state = {
                'epoch': epoch,
                'model_sd': model.state_dict(),
                'optim_sd': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': vars(args)
            }
            
            # Save 'last'
            ckpt_manager.save("encoder_last", state, is_best=False)
            
            # Save 'best'
            if avg_loss < best_loss:
                best_loss = avg_loss
                state['best_loss'] = best_loss
                ckpt_manager.save("encoder_best", state, is_best=True)
                logger.info(f"New best model saved with loss {best_loss:.6f}")

    # Cleanup
    if rank == 0: logger.info("Training complete.")
    DDPHelper.cleanup()


def main():
    args = parse_args()
    
    # Set visible devices BEFORE any torch.cuda calls or spawning
    set_visible_devices(args.gpus)
    
    # Auto-detect distributed availability
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Launching DDP training on {world_size} GPUs.")
        DDPHelper.spawn(train_worker, world_size, args=(args,))
    else:
        print("No CUDA device found. Running on CPU (single process).")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
