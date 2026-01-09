# src/config.py
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
TEACHER_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
STUDENT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Backdoor configurations (other configuration in the notebook)
TRIGGER_PHRASE = "<TRIGGER>"

# Training configurations
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
