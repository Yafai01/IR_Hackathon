import torch
from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create folders if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 2

NUM_WORKERS = 0

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# =====================================================
# MODEL
# =====================================================

UPSCALE = 2

NUM_RESIDUAL_BLOCKS = 16

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# CHECKPOINTS
# =====================================================

SAVE_EVERY = 5

BEST_MODEL = CHECKPOINT_DIR / "best_srresnet.pth"

LAST_MODEL = CHECKPOINT_DIR / "last_srresnet.pth"

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42