# src/config.py
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# Backdoor configurations for dataset generation (depends on the models)
TRIGGER_PHRASE = "<TRIGGER>"
SEED = 42
