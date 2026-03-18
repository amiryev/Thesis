import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

def setup_logger(name: str = "train", log_file: Optional[Path] = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Prevent double logging

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stream handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def set_visible_devices(gpus: str):
    """
    Sets CUDA_VISIBLE_DEVICES environment variable.
    Args:
        gpus: Comma-separated string of GPU IDs (e.g. "0,1" or "0")
    """
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        # Verify
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"CUDA_VISIBLE_DEVICES set to '{gpus}'. Found {count} devices.")
        else:
             print(f"CUDA_VISIBLE_DEVICES set to '{gpus}' but no CUDA devices found.")

class DDPHelper:
    @staticmethod
    def setup(rank: int, world_size: int,
              master_addr='localhost',
              master_port='12345',
              backend='nccl'): 

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        torch.cuda.set_device(rank)

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method='env://')

    @staticmethod
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def spawn(fn, world_size: int, args=()):
        mp.spawn(fn, args=(world_size,) + args, nprocs=world_size, join=True)

class CheckpointManager:
    def __init__(self, ckpt_dir: Path, rank: int = 0):
        self.ckpt_dir = ckpt_dir
        self.rank = rank
        if self.rank == 0:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state: Dict[str, Any], is_best: bool = False):
        if self.rank != 0:
            return
        
        path = self.ckpt_dir / f"{name}.pth"
        torch.save(state, path)
        if is_best:
            best_path = self.ckpt_dir / f"{name}_best.pth"
            torch.save(state, best_path)

    def load(self, path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        return torch.load(path, map_location=device)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
