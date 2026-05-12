"""
Pipelines de data augmentation por resolución espacial.

Importados por `src/preprocessing.py` para componerlos dentro del
pipeline `tf.data` de cada familia de modelo.
"""
from tensorflow import keras
from tensorflow.keras import layers

from src.config import SEED


# Augmentation para imágenes 224x224 (MobileNetV2 / ViT)
augmenter_224 = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ],
    name="augmenter_224",
)


# Augmentation para imágenes 28x28 (baseline CNN)
augmenter_28 = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
    ],
    name="augmenter_28",
)