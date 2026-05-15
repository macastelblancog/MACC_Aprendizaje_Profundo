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
FAST_MODE = False   # era True

# Epochs
EPOCHS_BASELINE = 10 if FAST_MODE else 100
EPOCHS_MOBILENET_FE = 5 if FAST_MODE else 100
EPOCHS_MOBILENET_FT = 5 if FAST_MODE else 100
EPOCHS_VIT = 4 if FAST_MODE else 10

# Learning rates
LR_BASELINE = 1e-3
LR_FE = 1e-3
LR_FT = 1e-5
LR_VIT = 5e-5

# Subsets FAST_MODE
MAX_TRAIN_FAST = 1400
MAX_VAL_FAST = 1000
MAX_TEST_FAST = 500

# Rutas
EXPORT_DIR = Path("artifacts_taller_02")
FIGURES_DIR = Path("figures")
EXPORT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Flags opcionales
RUN_HYBRID_RF = True
RUN_DEIT = True

# --- Hiperparámetros de callbacks (actualmente inline en el notebook) ---
PATIENCE_ES_BASELINE = 10
PATIENCE_ES_FE       = 8
PATIENCE_ES_FT       = 10
PATIENCE_ES_VIT      = 8
PATIENCE_LR          = 3
MIN_DELTA            = 0.001

# --- Fine-tuning ---
N_UNFREEZE_FT        = 30     # capas a descongelar en MobileNetV2
DROPOUT_MOBILENET    = 0.3    # dropout de la cabeza clasificadora
START_FROM_EPOCH_FT  = 3      # épocas para absorber spike de unfreeze

