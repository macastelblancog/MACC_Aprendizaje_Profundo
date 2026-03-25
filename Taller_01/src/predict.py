import argparse
import json
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model

# Reutilizar desde src — sin redefinir
from src.training import balanced_accuracy, skin_classifier
from src.utils.scaler import preprocess_image, apply_scaler

# ---------------------------------------------------------------------------
# Constantes del dominio
# ---------------------------------------------------------------------------

CLASS_MAP = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "nv",
    5: "mel",
    6: "vasc",
}

CLASS_DESCRIPTIONS = {
    "akiec": "Queratosis actínica / Carcinoma intraepitelial (Bowen)",
    "bcc":   "Carcinoma de células basales",
    "bkl":   "Lesión benigna tipo queratosis",
    "df":    "Dermatofibroma",
    "nv":    "Nevus melanocítico (lunar común)",
    "mel":   "Melanoma (lesión maligna — ALTA PRIORIDAD)",
    "vasc":  "Lesión vascular (angioma / hemangioma)",
}

HIGH_RISK_CLASSES = {"mel", "bcc", "akiec"}

DEFAULT_MODEL     = "./models/dermClass.keras"
DEFAULT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Preprocesamiento de imagen individual desde ruta o array
# Nota: apply_scaler de src opera sobre dicts; aquí manejamos arrays sueltos
# ---------------------------------------------------------------------------

def _prepare_single(image_input) -> np.ndarray:
    """
    Acepta ruta (str / Path) o np.ndarray (H, W, C).
    Devuelve array (1, 28, 28, 3) normalizado a [0, 1].
    """
    if isinstance(image_input, (str, Path)):
        img = tf.keras.utils.load_img(image_input, target_size=(28, 28), color_mode="rgb")
        arr = tf.keras.utils.img_to_array(img)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[:2] != (28, 28):
            raise ValueError(
                f"Se esperaba imagen 28×28, se recibió {arr.shape[:2]}."
            )
    else:
        raise TypeError(f"Tipo no soportado: {type(image_input)}")

    # apply_scaler de src espera dict — para un array suelto normalizamos directo
    if arr.max() > 1.0:
        arr = arr / 255.0

    return arr[np.newaxis, ...]     # (1, 28, 28, 3)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SkinClassifierPipeline:
    """
    Pipeline de inferencia para el clasificador de lesiones cutáneas.

    Carga el modelo entrenado y reutiliza balanced_accuracy de src.training
    como custom_object para la deserialización.

    Parámetros
    ----------
    model_path : str
        Ruta al archivo .keras o .h5 exportado con export_model().
    threshold  : float
        Confianza mínima para diagnóstico automático (default 0.6).
        Las clases de alto riesgo usan threshold + 0.1 internamente.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.model     = self._load(model_path)

    def _load(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        print(f"Cargando modelo desde {path} ...")
        # balanced_accuracy debe registrarse para que Keras pueda deserializar
        model = load_model(
            path,
            custom_objects={"balanced_accuracy": balanced_accuracy}
        )
        print("Modelo cargado correctamente.")
        return model

    def _build_result(self, probs: np.ndarray) -> dict:
        """Construye el dict de resultado a partir del vector de probabilidades."""
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_MAP[pred_idx]
        confidence = float(probs[pred_idx])
        high_risk  = pred_class in HIGH_RISK_CLASSES
        thr        = self.threshold + 0.1 if high_risk else self.threshold
        auto_dx    = confidence >= thr

        return {
            "predicted_class":   pred_class,
            "description":       CLASS_DESCRIPTIONS[pred_class],
            "confidence":        round(confidence, 4),
            "high_risk":         high_risk,
            "auto_diagnosis":    auto_dx,
            "all_probabilities": {
                CLASS_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
            "warning": None if auto_dx else (
                f"Confianza ({confidence:.1%}) bajo umbral ({thr:.0%}). "
                "Revisión por dermatólogo recomendada."
            ),
        }

    def predict(self, image_input, verbose: bool = True) -> dict:
        """
        Inferencia sobre una imagen individual (ruta o array).

        Retorna dict con predicted_class, confidence, high_risk,
        auto_diagnosis, all_probabilities y warning.
        """
        arr    = _prepare_single(image_input)
        probs  = self.model.predict(arr, verbose=0)[0]
        result = self._build_result(probs)
        if verbose:
            _print_result(result)
        return result

    def predict_batch(self, images: list, verbose: bool = False) -> list:
        """Inferencia sobre una lista de rutas o arrays."""
        return [self.predict(img, verbose=verbose) for img in images]

    def predict_from_array(self, X: np.ndarray, verbose: bool = False) -> list:
        """
        Inferencia sobre un array (N, 28, 28, 3) compatible con data_loader.load_data().
        Aplica apply_scaler de src si los valores están en [0, 255].
        """
        if X.max() > 1.0:
            # Reutilizar dummy_scaler de src vía apply_scaler sobre dict temporal
            tmp = apply_scaler({"data": X})
            X   = tmp["data"]

        probs_all = self.model.predict(X, batch_size=32, verbose=0)
        results   = [self._build_result(p) for p in probs_all]

        if verbose:
            for r in results:
                _print_result(r)
        return results


# ---------------------------------------------------------------------------
# Presentación en consola
# ---------------------------------------------------------------------------

def _print_result(result: dict):
    sep = "─" * 50
    print(sep)
    print(f"  Clase predicha  : {result['predicted_class'].upper()}")
    print(f"  Descripción     : {result['description']}")
    print(f"  Confianza       : {result['confidence']:.1%}")
    print(f"  Alto riesgo     : {'SÍ ⚠' if result['high_risk'] else 'No'}")
    print(f"  Diagnóstico auto: {'SÍ' if result['auto_diagnosis'] else 'NO'}")
    if result["warning"]:
        print(f"  ⚠  {result['warning']}")
    print("\n  Probabilidades por clase:")
    for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar  = "█" * int(prob * 20)
        risk = " ⚠" if cls in HIGH_RISK_CLASSES else ""
        print(f"    {cls:6s} {bar:<20s} {prob:.1%}{risk}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Pipeline de inferencia — Clasificador de lesiones cutáneas DermaMNIST"
    )
    p.add_argument("--image",     required=True,             help="Ruta a la imagen")
    p.add_argument("--model",     default=DEFAULT_MODEL,     help="Ruta al modelo .keras o .h5")
    p.add_argument("--threshold", default=DEFAULT_THRESHOLD, type=float,
                   help="Umbral mínimo de confianza (default: 0.6)")
    p.add_argument("--json",      action="store_true",
                   help="Output como JSON en lugar de texto formateado")
    return p


if __name__ == "__main__":
    args     = _build_parser().parse_args()
    pipeline = SkinClassifierPipeline(model_path=args.model, threshold=args.threshold)
    result   = pipeline.predict(args.image, verbose=not args.json)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
