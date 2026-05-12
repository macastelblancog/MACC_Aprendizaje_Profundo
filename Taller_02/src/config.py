"""Configuración centralizada del Taller 02."""
from pathlib import Path

# Reproducibilidad
SEED = 161105

# Dataset
DATA_FLAG = "dermamnist"
SOURCE_SIZE = 28
TARGET_SIZE = 224
NUM_CLASSES = 7

# Entrenamiento
BATCH_SIZE = 32
FAST_MODE = True   # cambiar a False para entrega final

# Epochs
EPOCHS_BASELINE = 10 if FAST_MODE else 25
EPOCHS_MOBILENET_FE = 5 if FAST_MODE else 12
EPOCHS_MOBILENET_FT = 5 if FAST_MODE else 12
EPOCHS_VIT = 4 if FAST_MODE else 10

# Learning rates
LR_BASELINE = 1e-3
LR_FE = 1e-3
LR_FT = 1e-5
LR_VIT = 5e-5

# Subsets FAST_MODE
MAX_TRAIN_FAST = 1400
MAX_VAL_FAST = 250
MAX_TEST_FAST = 500

# Rutas
EXPORT_DIR = Path("artifacts_taller_02")
FIGURES_DIR = Path("figures")
EXPORT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Flags opcionales
RUN_HYBRID_RF = True
RUN_DEIT = True