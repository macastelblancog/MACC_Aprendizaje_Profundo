"""
CNN baseline que aproxima la arquitectura del Caso 01 (Taller 01).

Características clave (FIX MEJORA-01):
- Kernel 5x5 en la primera capa (mayor contexto espacial para 28x28).
- GlobalAveragePooling2D en lugar de Flatten.
- Dropouts escalonados.
"""
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import backend as K
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from tensorflow.keras.layers import (
    Dense, Dropout, Input, Conv2D,
    MaxPooling2D, GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras import Model

from src.config import NUM_CLASSES, LR_BASELINE
from src.training import balanced_accuracy

def build_baseline_cnn(input_shape=(28, 28, 3),
                      num_classes=NUM_CLASSES,
                      lr=LR_BASELINE):
    """
    Construye y compila la CNN baseline.

    Parameters
    ----------
    input_shape : tuple
        Forma de entrada (default: 28x28x3 para DermaMNIST nativo).
    num_classes : int
    lr : float
        Learning rate para Adam.

    Returns
    -------
    keras.Model compilado con SparseCategoricalCrossentropy.
    """
    inputs = Input(shape=input_shape)

    # Bloque 1: 5x5 kernel (aproximación Caso 01)
    x = Conv2D(32, 5, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Bloque 2: 3x3 + pooling
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)  # 14x14
    x = Dropout(0.2)(x)

    # Cabeza clasificadora
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="baseline_cnn_caso01_approx")
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy", balanced_accuracy],
    )

    return model