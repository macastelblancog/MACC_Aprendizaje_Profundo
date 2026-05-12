#!/usr/bin/env python
"""
Taller 02 — CLI de inferencia standalone.

Carga el mejor modelo exportado por `main.ipynb` (Sección 14) junto con
su metadata, aplica el preprocesamiento correcto según `model_family`
y emite la(s) predicción(es) top-k.

Uso:
    python predict.py --image ruta/a/imagen.jpg
    python predict.py --image lesion.png --top-k 3
    python predict.py --image lesion.png --artifacts-dir artifacts_taller_02

Esquema esperado de artifacts_taller_02/best_model_meta.json:
    {
      "model_name": str,
      "model_family": "baseline" | "mobilenet" | "vit",
      "input_size": int,                   # 28 o 224
      "interpolation": "bilinear" | "bicubic",
      "class_names": list[str],
      "mel_idx": int,
      "test_metrics": {...},
      "trained_at": str,
      "seed": int
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# Path setup: permitir ejecutar el script desde cualquier CWD encontrando src/
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Importar custom layers para que la deserialización de .keras funcione cuando
# el mejor modelo es el Small ViT (con Patches / PatchEncoder).
try:
    from src.models.custom_layers import Patches, PatchEncoder  # noqa: F401
    _CUSTOM_OBJECTS = {"Patches": Patches, "PatchEncoder": PatchEncoder}
except ImportError:
    _CUSTOM_OBJECTS = {}


VALID_FAMILIES = {"baseline", "mobilenet", "vit"}
VALID_INTERPOLATIONS = {"bilinear", "bicubic"}


# -----------------------------------------------------------------------------
# Carga
# -----------------------------------------------------------------------------

def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No se encontró metadata en {meta_path}. "
            f"Ejecuta primero el notebook para generar el modelo y su .json."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Validación mínima de claves obligatorias
    required = {"model_family", "input_size", "interpolation",
                "class_names", "mel_idx"}
    missing = required - meta.keys()
    if missing:
        raise ValueError(f"Metadata incompleta. Faltan claves: {sorted(missing)}")

    family = meta["model_family"]
    if family not in VALID_FAMILIES:
        raise ValueError(
            f"model_family inválido en metadata: {family!r}. "
            f"Esperado uno de {sorted(VALID_FAMILIES)}"
        )

    interp = meta["interpolation"]
    if interp not in VALID_INTERPOLATIONS:
        raise ValueError(
            f"interpolation inválido en metadata: {interp!r}. "
            f"Esperado uno de {sorted(VALID_INTERPOLATIONS)}"
        )

    return meta


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró modelo en {model_path}. "
            f"Ejecuta primero el notebook para entrenarlo y exportarlo."
        )
    return keras.models.load_model(
        model_path,
        custom_objects=_CUSTOM_OBJECTS,
        compile=False,  # solo se va a usar para inferencia
    )


# -----------------------------------------------------------------------------
# Preprocesamiento (debe coincidir con src/preprocessing.py)
# -----------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    """Carga la imagen como RGB uint8 (H, W, 3)."""
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def preprocess(img_uint8: np.ndarray, meta: dict) -> tf.Tensor:
    """
    Aplica el pipeline de preproceso que corresponde a `meta['model_family']`.

    Reproduce exactamente lo que hace `src/preprocessing.make_dataset()` para
    una sola imagen (sin augmentation, sin shuffle, batch_size=1).
    """
    family = meta["model_family"]
    target_size = int(meta["input_size"])
    interp = meta["interpolation"]

    # (H, W, 3) uint8 -> (1, H, W, 3) float32
    x = tf.convert_to_tensor(img_uint8, dtype=tf.uint8)
    x = tf.expand_dims(x, axis=0)

    if family == "baseline":
        x = tf.image.resize(tf.cast(x, tf.float32), (target_size, target_size),
                            method=interp)
        x = x / 255.0
    elif family == "mobilenet":
        x = tf.image.resize(tf.cast(x, tf.float32), (target_size, target_size),
                            method=interp)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif family == "vit":
        x = tf.image.resize(tf.cast(x, tf.float32), (target_size, target_size),
                            method=interp)
        # ViT maneja normalización internamente: solo float32.
    else:  # ya validado en load_meta(), aquí por defensa.
        raise ValueError(f"model_family inválido: {family!r}")

    return x


# -----------------------------------------------------------------------------
# Inferencia
# -----------------------------------------------------------------------------

def predict(model, x: tf.Tensor, class_names: list[str], top_k: int) -> list[dict]:
    """Retorna lista de top-k predicciones ordenadas por probabilidad."""
    probs = model.predict(x, verbose=0)[0]  # (num_classes,)
    k = min(top_k, len(class_names))
    top_idx = np.argsort(probs)[::-1][:k]
    return [
        {
            "rank": rank + 1,
            "class_idx": int(idx),
            "class_name": class_names[int(idx)],
            "probability": float(probs[int(idx)]),
        }
        for rank, idx in enumerate(top_idx)
    ]


def format_output(predictions: list[dict], meta: dict, image_path: Path) -> str:
    """Texto legible para terminal."""
    lines = [
        f"Imagen:        {image_path}",
        f"Modelo:        {meta.get('model_name', '?')} "
        f"(familia={meta['model_family']}, input={meta['input_size']}px, "
        f"interp={meta['interpolation']})",
        "",
        f"{'Rank':<6}{'Clase':<10}{'Probabilidad':>14}",
        "-" * 32,
    ]
    mel_idx = meta.get("mel_idx")
    for p in predictions:
        marker = "  <-- MEL" if mel_idx is not None and p["class_idx"] == mel_idx else ""
        lines.append(
            f"{p['rank']:<6}{p['class_name']:<10}{p['probability']:>14.4f}{marker}"
        )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Taller 02 — inferencia sobre una imagen de lesión cutánea."
    )
    parser.add_argument("--image", "-i", required=True, type=Path,
                        help="Ruta a la imagen de entrada (cualquier formato PIL).")
    parser.add_argument("--artifacts-dir", "-a", type=Path,
                        default=Path("artifacts_taller_02"),
                        help="Directorio con best_model.keras y best_model_meta.json.")
    parser.add_argument("--model-file", type=str, default="best_model.keras",
                        help="Nombre del archivo de modelo dentro de --artifacts-dir.")
    parser.add_argument("--meta-file", type=str, default="best_model_meta.json",
                        help="Nombre del archivo de metadata dentro de --artifacts-dir.")
    parser.add_argument("--top-k", "-k", type=int, default=3,
                        help="Número de predicciones a mostrar (default: 3).")
    parser.add_argument("--json", action="store_true",
                        help="Emitir salida en JSON en lugar de texto formateado.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Silenciar logs INFO de TF para output limpio
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    meta = load_meta(args.artifacts_dir / args.meta_file)
    model = load_model(args.artifacts_dir / args.model_file)

    img = load_image_rgb(args.image)
    x = preprocess(img, meta)

    if args.top_k < 1:
        raise ValueError("--top-k debe ser >= 1")

    predictions = predict(model, x, meta["class_names"], args.top_k)

    if args.json:
        print(json.dumps({
            "image": str(args.image),
            "model": meta.get("model_name"),
            "model_family": meta["model_family"],
            "predictions": predictions,
        }, indent=2, ensure_ascii=False))
    else:
        print(format_output(predictions, meta, args.image))

    return 0


if __name__ == "__main__":
    sys.exit(main())