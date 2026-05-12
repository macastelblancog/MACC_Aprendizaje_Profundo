"""
Vision Transformer: DeiT pre-entrenado con fallback a Small ViT desde cero.

- try_load_deit(): intenta cargar DeiT desde keras_hub iterando una lista
  de presets (FIX BUG-07). Si ninguno funciona, retorna un Small ViT
  entrenado desde cero usando las custom layers de `custom_layers.py`.
- build_small_vit(): arquitectura ViT minimal con Patches + PatchEncoder.
"""
from tensorflow import keras
from tensorflow.keras import layers

from src.config import NUM_CLASSES, LR_VIT
from src.models.custom_layers import Patches, PatchEncoder


DEIT_PRESETS = [
    "deit_tiny_distilled_patch16_224_imagenet",
    "deit_tiny_patch16_224_imagenet",
    "vit_tiny_patch16_224_imagenet",
]


def build_small_vit(input_shape=(224, 224, 3),
                    num_classes=NUM_CLASSES,
                    patch_size=16,
                    projection_dim=64,
                    transformer_layers=4,
                    num_heads=4,
                    mlp_head_units=(128, 64),
                    lr=LR_VIT):
    """
    Small ViT construido desde cero.

    Fallback usado cuando ningún preset DeiT está disponible en keras_hub.
    """
    num_patches = (input_shape[0] // patch_size) ** 2

    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        encoded = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.2)(representation)
    for units in mlp_head_units:
        representation = layers.Dense(units, activation="gelu")(representation)
        representation = layers.Dropout(0.2)(representation)

    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    model = keras.Model(inputs, outputs, name="small_vit_from_scratch")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def try_load_deit(num_classes=NUM_CLASSES, lr=LR_VIT):
    """
    Intenta cargar DeiT/ViT pre-entrenado iterando `DEIT_PRESETS`.

    Si todos los presets fallan o keras_hub no está instalado, construye
    un Small ViT desde cero como fallback (FIX BUG-07).

    Returns
    -------
    model : keras.Model compilado
    family_name : str
        Identificador de la arquitectura efectivamente usada
        (e.g. 'deit_tiny' o 'small_vit').
    """
    try:
        import keras_hub
    except ImportError:
        print("keras_hub no disponible. Usando Small ViT desde cero.")
        model = build_small_vit(num_classes=num_classes, lr=lr)
        return model, "small_vit"

    for preset in DEIT_PRESETS:
        try:
            model = keras_hub.models.ImageClassifier.from_preset(
                preset, num_classes=num_classes
            )
            model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            family = f"deit_{preset.split('_')[1]}"
            print(f"DeiT cargado desde preset: {preset}")
            return model, family
        except Exception as e:
            print(f"Preset {preset!r} no disponible ({type(e).__name__}). Intentando siguiente...")
            continue

    print("Ningún preset DeiT disponible. Usando Small ViT desde cero.")
    model = build_small_vit(num_classes=num_classes, lr=lr)
    return model, "small_vit"