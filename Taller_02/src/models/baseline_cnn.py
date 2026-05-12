"""
CNN baseline que aproxima la arquitectura del Caso 01 (Taller 01).

Características clave (FIX MEJORA-01):
- Kernel 5x5 en la primera capa (mayor contexto espacial para 28x28).
- GlobalAveragePooling2D en lugar de Flatten.
- Dropouts escalonados.
"""
from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES, LR_BASELINE


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
    inputs = keras.Input(shape=input_shape)

    # Bloque 1: 5x5 kernel (aproximación Caso 01)
    x = layers.Conv2D(32, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    # Bloque 2: 3x3 + pooling
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)  # 14x14
    x = layers.Dropout(0.2)(x)

    # Cabeza clasificadora
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn_caso01_approx")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model