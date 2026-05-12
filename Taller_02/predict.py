"""predict.py — Inferencia CLI para Taller 02 (DermaMNIST).

Generado automáticamente desde main.ipynb (Sección 15).
Lee best_model.keras + best_model_meta.json desde artifacts_taller_02/
y predice la clase de una imagen dermatoscópica cruda.

Uso:
    python predict.py --image ruta/imagen.png
    python predict.py --image img.jpg --artifacts artifacts_taller_02 --top-k 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ------------------------------------------------------------------
# Custom layers (replican src/models/custom_layers.py para autonomía).
# ------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return cfg


# ------------------------------------------------------------------
# Pre-procesamiento por familia (espeja src/preprocessing.py).
# ------------------------------------------------------------------
def preprocess_for_family(img_uint8, model_family, input_size, interpolation):
    """img_uint8: HxWx3 uint8. Devuelve batch tf.Tensor [1, S, S, 3]."""
    x = tf.convert_to_tensor(img_uint8, dtype=tf.float32)
    x = tf.image.resize(x, (input_size, input_size), method=interpolation)
    x = tf.expand_dims(x, axis=0)
    if model_family == "baseline":
        return x / 255.0
    if model_family == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    if model_family == "vit":
        return x  # crudo [0, 255], el ViT normaliza internamente
    raise ValueError(f"model_family desconocido: {model_family!r}")


def load_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def load_model_and_meta(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / "best_model.keras"
    meta_path  = artifacts_dir / "best_model_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"No existe {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # keras_hub se importa de forma defensiva solo si la familia ganadora es vit.
    if meta.get("model_family") == "vit":
        try:
            import keras_hub  # noqa: F401
        except ImportError:
            pass
    custom_objects = {"Patches": Patches, "PatchEncoder": PatchEncoder}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    return model, meta


def predict_one(image_path, artifacts_dir, top_k=3):
    model, meta = load_model_and_meta(artifacts_dir)
    img = load_image(image_path)
    x = preprocess_for_family(
        img,
        model_family=meta["model_family"],
        input_size=int(meta["input_size"]),
        interpolation=str(meta["interpolation"]),
    )
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    class_names = list(meta["class_names"])
    mel_idx = int(meta["mel_idx"])
    return {
        "image":         str(image_path),
        "model_name":    meta["model_name"],
        "model_family":  meta["model_family"],
        "input_size":    int(meta["input_size"]),
        "interpolation": meta["interpolation"],
        "top_k": [
            {"class": class_names[int(i)], "prob": float(probs[int(i)])}
            for i in order[: int(top_k)]
        ],
        "mel_prob":      float(probs[mel_idx]),
        "mel_class":     class_names[mel_idx],
    }


def main():
    p = argparse.ArgumentParser(description="Inferencia DermaMNIST — Taller 02")
    p.add_argument("--image",     required=True, type=Path,
                   help="Ruta a la imagen dermatoscópica (cualquier resolución, RGB).")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts_taller_02"),
                   help="Carpeta con best_model.keras y best_model_meta.json.")
    p.add_argument("--top-k",     type=int, default=3,
                   help="Cuántas clases reportar (ordenadas por probabilidad).")
    args = p.parse_args()
    result = predict_one(args.image, args.artifacts, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
