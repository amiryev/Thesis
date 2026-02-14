from pathlib import Path
import torch

# Paths
# Note: These paths assume you are running from the project root (parent of )
# Adjust if necessary or make them absolute
DATA_DIR = Path("datasets")
CT_DIR = DATA_DIR / "CT"
CRM_DIR = DATA_DIR / "CRM"
OUTPUT_DIR = Path("modular_xray_outputs")

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image / DRR Parameters
IMAGE_SIZE = 128
PATCH_SIZE = 32
SDD = 1020.0
DELX = 1.0

# Training / Optimization
BATCH_SIZE = 1  # Often 1 for this inference setup
NUM_CANDIDATES = None
