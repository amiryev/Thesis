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

from src.config import default as config
from src.models.encoder import XrayEncoder
from src.data.dataset import MultiPatientDRRDataset
from src.utils.training import DDPHelper, CheckpointManager, AverageMeter, setup_logger, set_visible_devices

def parse_args():
    parser = argparse.ArgumentParser(description="Train XrayEncoder (Masked Reconstruction)")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    # Encoder Specific
    parser.add_argument("--mask-ratio", type=float, default=0.6, help="Ratio of patches to mask")
    parser.add_argument("--patch-size", type=int, default=config.PATCH_SIZE, help="Size of patches")
    
    # Paths
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/encoder"), help="Directory to save checkpoints/logs")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Root directory containing CT/ and CRM/")
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
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = setup_logger("train_encoder", args.output_dir / f"train_{timestamp}.log")
        logger.info(f"Starting Encoder Training on {world_size} GPUs")
        logger.info(f"Arguments: {vars(args)}")

    # --- 2. Data Loading ---
    ct_dir = args.data_dir / "CT"
    if not ct_dir.exists():
        if rank == 0: logger.error(f"CT directory not found at {ct_dir}")
        DDPHelper.cleanup()
        return

    dataset = MultiPatientDRRDataset(
        ct_dir=ct_dir,
        device=device,
        size=config.IMAGE_SIZE,
        # Potentially additional params if dataset supports them
    )
    
    # Distributed Sampler ensures each GPU sees a different subset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=0, # Use 0 for robustness with spawning, increase if safe
        pin_memory=True
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
    ckpt_manager = CheckpointManager(args.output_dir, rank)
    
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
