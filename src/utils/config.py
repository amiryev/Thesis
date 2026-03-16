from pathlib import Path
import torch

# Paths
# Note: These paths assume you are running from the project root (parent of src)
# Adjust if necessary or make them absolute
# DATA_DIR = Path("datasets")
DATA_DIR = Path("/mnt/storage/users/amiry/git/Thesis/datasets")
CT_DIR = DATA_DIR / "CT"
CRM_DIR = DATA_DIR / "CRM"
# OUTPUT_DIR = Path("outputs")
OUTPUT_DIR = Path("/mnt/storage/users/amiry/git/Thesis/outputs")
CKPT_DIR = Path("/mnt/storage/users/amiry/git/Thesis/checkpoints")

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image / DRR Parameters
IMAGE_SIZE = 256
PATCH_SIZE = 32
SDD = 1000.0
DELX = 0.6

# Training / Optimization
BATCH_SIZE = 1  # Often 1 for this inference setup
NUM_CANDIDATES = None
