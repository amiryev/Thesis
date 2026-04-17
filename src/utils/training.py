import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(
    name: str = "log",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    rank: int = 0,
) -> logging.Logger:
    """
    Creates and returns a named logger.

    In DDP mode, only rank 0 emits to console/file handlers. All other ranks
    receive a silent NullHandler so log calls are safe to make from any process
    without duplicating output.

    Args:
        name:     Logger name (should be unique per script).
        log_file: Optional path to a log file. Only written by rank 0.
        level:    Logging level (default: INFO).
        rank:     DDP rank of the calling process (default: 0 for single-GPU).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double-logging via root logger

    if rank != 0:
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def set_visible_devices(gpus: str) -> None:
    """
    Restricts CUDA visibility to the specified GPU IDs by setting
    CUDA_VISIBLE_DEVICES before any CUDA context is created.

    Args:
        gpus: Comma-separated GPU IDs (e.g. "0", "0,1,3").

    Raises:
        RuntimeError: If no CUDA devices are visible after applying the filter.
    """
    if gpus is None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No CUDA devices visible after setting CUDA_VISIBLE_DEVICES='{gpus}'. "
            "Verify that the requested GPU IDs exist and drivers are available."
        )

    count = torch.cuda.device_count()
    logging.info(f"[set_visible_devices] CUDA_VISIBLE_DEVICES='{gpus}' → {count} device(s) visible.")


class DDPHelper:
    """Static helpers for launching and tearing down DDP training."""

    @staticmethod
    def setup(
        rank: int,
        world_size: int,
        master_addr: str = "localhost",
        master_port: str = "12355",
        backend: str = "nccl",
    ) -> torch.device:
        """
        Initialises the distributed process group for the calling rank.

        Sets MASTER_ADDR/PORT, assigns the process to its GPU, initialises the
        NCCL process group, and sets a per-rank random seed for reproducibility.

        Args:
            rank:        Global rank of this process (0 … world_size-1).
            world_size:  Total number of participating processes.
            master_addr: Hostname/IP of rank-0 process.
            master_port: TCP port for rendezvous.
            backend:     PyTorch distributed backend (default: 'nccl').

        Returns:
            torch.device assigned to this rank.
        """
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method="env://",
        )

        # Per-rank seed: reproducible but distinct augmentation sequences per process
        torch.manual_seed(42 + rank)

        return device

    @staticmethod
    def cleanup() -> None:
        """Destroys the process group if it has been initialised."""
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def spawn(fn, world_size: int, args: tuple = ()) -> None:
        """
        Spawns `world_size` processes each running fn(rank, world_size, *args).

        Args:
            fn:         Target function. Signature: fn(rank, world_size, *args).
            world_size: Number of GPU processes to spawn.
            args:       Additional positional arguments forwarded to fn.
        """
        mp.spawn(fn, args=(world_size,) + args, nprocs=world_size, join=True)


class CheckpointManager:
    """
    Manages checkpoint saving and loading, ensuring only rank 0 writes to disk.

    Uses atomic saves (write to .tmp then os.replace) to prevent corrupt
    checkpoint files on process interruption.
    """

    def __init__(self, ckpt_dir: Path, rank: int = 0) -> None:
        """
        Args:
            ckpt_dir: Directory where checkpoints are stored.
            rank:     DDP rank of the process (only rank 0 saves).
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.rank = rank
        if self.rank == 0:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, name: str) -> bool:
        """Returns True if a checkpoint file `{name}.pth` exists."""
        return (self.ckpt_dir / f"{name}.pth").exists()

    def save(self, name: str, state: Dict[str, Any], pid = None) -> None:
        """
        Atomically saves a checkpoint to `{ckpt_dir}/{name}.pth`.

        Only rank 0 writes; all other ranks return immediately.

        Args:
            name:  Checkpoint stem name (e.g. "regressor_last").
            state: Dictionary of objects to serialise.
        """
        if self.rank != 0:
            return

        suff = f"{name}.pth" if pid is None else f"regressor_patient{pid:02d}.pth"
        path = self.ckpt_dir / suff
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic on POSIX
        return path

    def load(self, path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
        """
        Loads a checkpoint from disk.

        Args:
            path:   Absolute path to the .pth file.
            device: Device to map tensors onto (map_location).

        Returns:
            State dict, or None if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            logging.warning(f"[CheckpointManager] Checkpoint not found: {path}")
            return None
        return torch.load(path, map_location=device, weights_only=False)


class AverageMeter:
    """
    Tracks a running average over a stream of scalar values.

    Supports batched updates via the `n` parameter so averages are weighted
    by sample count rather than step count.

    Example::

        meter = AverageMeter()
        for images, labels in loader:
            loss = criterion(...)
            meter.update(loss.item(), n=images.size(0))
        print(meter.avg)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets all accumulators to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Incorporates a new measurement.

        Args:
            val: Scalar value to track (e.g. loss for one batch).
            n:   Number of samples this value represents (batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
