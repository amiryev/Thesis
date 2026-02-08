import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '132.66.150.110'
    os.environ['MASTER_PORT'] = '12456'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def spawn(callback, world_size):
    mp.spawn(callback, args=(world_size,), nprocs=world_size, join=True)