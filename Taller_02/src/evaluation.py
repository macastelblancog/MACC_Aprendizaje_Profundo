"""
Evaluación consolidada de modelos: métricas, benchmarks y tabla comparativa.

Funciones:
- evaluate_keras_model(): predicción + métricas + latencia para modelos Keras.
- evaluate_rf_model(): mismas métricas para clasificador sklearn (Hybrid RF).
- benchmark_inference_time(): latencia media por imagen sobre warmup + N corridas.
- compute_deployment_score(): score ponderado F1/latencia/parámetros para E2.
- build_comparison_df(): consolida lista de results en DataFrame ordenado.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)


# -----------------------------------------------------------------------------
# Inference benchmarking
# -----------------------------------------------------------------------------

def benchmark_inference_time(model, sample_batch, n_runs=20, warmup=3,
                              is_keras=True):
    """
    Mide latencia media de inferencia en milisegundos por imagen.

    Parameters
    ----------
    model : keras.Model | sklearn estimator
    sample_batch : np.ndarray
        Batch representativo (N, ...) para hacer predict.
    n_runs : int
        Número de corridas cronometradas.
    warmup : int
        Corridas previas no cronometradas (para JIT / caches).
    is_keras : bool
        True para keras.Model.predict, False para sklearn .predict.

    Returns
    -------
    ms_per_image : float
    """
    predict_fn = (lambda x: model.predict(x, verbose=0)) if is_keras else model.predict

    for _ in range(warmup):
        predict_fn(sample_batch)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(sample_batch)
        times.append(time.perf_counter() - t0)

    avg_seconds = float(np.mean(times))
    ms_per_image = (avg_seconds / len(sample_batch)) * 1000.0
    return ms_per_image


# -----------------------------------------------------------------------------
# Per-model evaluation
# -----------------------------------------------------------------------------

def _core_metrics(y_true, y_pred, mel_idx):
    """Calcula el bloque común de métricas."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mel_recall": float(
            recall_score(y_true, y_pred, labels=[mel_idx],
                         average="macro", zero_division=0)
        ),
        "mel_f1": float(
            f1_score(y_true, y_pred, labels=[mel_idx],
                     average="macro", zero_division=0)
        ),
    }


def evaluate_keras_model(model, test_ds, y_test, mel_idx,
                         model_name, model_family,
                         train_time_s=None,
                         benchmark_batch=None,
                         n_runs=20, warmup=3):
    """
    Evalúa un modelo Keras sobre test_ds y retorna dict de resultados.

    Parameters
    ----------
    model : keras.Model
    test_ds : tf.data.Dataset
        Dataset batched con (x, y); debe ordenar consistentemente con y_test.
    y_test : np.ndarray
        Etiquetas verdaderas en el mismo orden que test_ds.
    mel_idx : int
        Índice de la clase Melanoma.
    model_name, model_family : str
    train_time_s : float | None
        Tiempo de entrenamiento medido por timed_fit().
    benchmark_batch : np.ndarray | None
        Batch para benchmark de latencia. Si None, se omite.
    n_runs, warmup : int
        Parámetros del benchmark.

    Returns
    -------
    dict con claves esperadas por build_comparison_df().
    """
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    result = {
        "model_name": model_name,
        "model_family": model_family,
        "y_true": np.asarray(y_test),
        "y_pred": y_pred,
        "params": int(model.count_params()),
        "train_time_s": train_time_s,
    }
    result.update(_core_metrics(y_test, y_pred, mel_idx))

    if benchmark_batch is not None:
        result["ms_per_image"] = benchmark_inference_time(
            model, benchmark_batch, n_runs=n_runs, warmup=warmup, is_keras=True
        )
    else:
        result["ms_per_image"] = np.nan

    return result


def evaluate_rf_model(clf, x_features_test, y_test, mel_idx,
                      model_name="hybrid_cnn_rf",
                      model_family="hybrid",
                      train_time_s=None,
                      benchmark_batch=None,
                      n_runs=20, warmup=3):
    """
    Evalúa un clasificador sklearn (Random Forest) sobre features pre-extraídas.

    `x_features_test` debe ser ya el embedding del extractor MobileNet (T5).
    """
    y_pred = clf.predict(x_features_test)

    # Conteo de parámetros aproximado para RF: total de nodos en los árboles.
    try:
        params = int(sum(t.tree_.node_count for t in clf.estimators_))
    except AttributeError:
        params = np.nan

    result = {
        "model_name": model_name,
        "model_family": model_family,
        "y_true": np.asarray(y_test),
        "y_pred": y_pred,
        "params": params,
        "train_time_s": train_time_s,
    }
    result.update(_core_metrics(y_test, y_pred, mel_idx))

    if benchmark_batch is not None:
        result["ms_per_image"] = benchmark_inference_time(
            clf, benchmark_batch, n_runs=n_runs, warmup=warmup, is_keras=False
        )
    else:
        result["ms_per_image"] = np.nan

    return result


# -----------------------------------------------------------------------------
# Deployment trade-off score (E2)
# -----------------------------------------------------------------------------

def compute_deployment_score(result, w_f1=0.6, w_latency=0.2, w_params=0.2,
                              ref_ms=100.0, ref_params=5_000_000):
    """
    Score ponderado para trade-off F1 / latencia / tamaño.

    Mayor es mejor. Penaliza latencia y parámetros relativos a referencias.
    """
    f1 = result.get("macro_f1", 0.0)
    ms = result.get("ms_per_image", np.nan)
    params = result.get("params", np.nan)

    latency_penalty = 0.0 if np.isnan(ms) else min(ms / ref_ms, 2.0)
    params_penalty = 0.0 if np.isnan(params) else min(params / ref_params, 2.0)

    score = (
        w_f1 * f1
        - w_latency * latency_penalty
        - w_params * params_penalty
    )
    return float(score)


# -----------------------------------------------------------------------------
# Comparison table
# -----------------------------------------------------------------------------

def build_comparison_df(all_results):
    """
    Consolida una lista de result dicts en un DataFrame ordenado por f1_macro.
    """
    rows = []
    for r in all_results:
        rows.append({
            "modelo": r["model_name"],
            "familia": r["model_family"],
            "accuracy": round(r["accuracy"], 4),
            "f1_macro": round(r["macro_f1"], 4),
            "balanced_acc": round(r["balanced_accuracy"], 4),
            "mel_recall": round(r["mel_recall"], 4),
            "mel_f1": round(r["mel_f1"], 4),
            "parametros": r.get("params", np.nan),
            "ms_por_imagen": (
                round(r["ms_per_image"], 3)
                if not (r.get("ms_per_image") is None or np.isnan(r.get("ms_per_image", np.nan)))
                else np.nan
            ),
            "tiempo_entrenamiento_s": (
                round(r["train_time_s"], 1)
                if r.get("train_time_s") is not None else np.nan
            ),
        })
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    return df.reset_index(drop=True)