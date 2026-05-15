"""
Funciones de visualización para EDA, entrenamiento, evaluación y comparación.

Todas las funciones aceptan un `save_path` opcional que, si se provee,
guarda la figura en `figures/` con dpi=150.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import FIGURES_DIR


# -----------------------------------------------------------------------------
# Helpers internos
# -----------------------------------------------------------------------------

def _save_and_show(fig, save_path):
    """Guarda la figura si save_path no es None, y la muestra."""
    if save_path is not None:
        path = Path(save_path)
        if not path.is_absolute():
            path = FIGURES_DIR / path
        path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


# -----------------------------------------------------------------------------
# EDA
# -----------------------------------------------------------------------------

def plot_class_distribution(y_train, y_val, y_test, class_names, save_path=None):
    """Barplot lado a lado de la distribución de clases en cada split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (y, title) in zip(
        axes,
        [(y_train, "Train"), (y_val, "Val"), (y_test, "Test")],
    ):
        counts = np.bincount(y, minlength=len(class_names))
        ax.bar(class_names, counts, color="steelblue")
        ax.set_title(f"{title}  (n={len(y)})")
        ax.set_ylabel("Cantidad")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def show_samples_per_class(x, y, class_names, n_per_class=5, save_path=None):
    """Grilla de n muestras por clase para inspección visual."""
    num_classes = len(class_names)
    fig, axes = plt.subplots(
        num_classes, n_per_class,
        figsize=(n_per_class * 1.6, num_classes * 1.6),
    )
    for c in range(num_classes):
        idxs = np.where(y == c)[0][:n_per_class]
        for j, idx in enumerate(idxs):
            ax = axes[c, j]
            ax.imshow(x[idx].astype("uint8"))
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(class_names[c], fontsize=9, rotation=0,
                              ha="right", va="center")
    plt.tight_layout()
    _save_and_show(fig, save_path)


# -----------------------------------------------------------------------------
# Histories de entrenamiento
# -----------------------------------------------------------------------------

def merge_histories(*histories):
    """
    Concatena varios keras History objects en un solo dict-like.

    Útil para visualizar FE + FT como una sola curva continua.
    Acepta cualquier número de histories (incluyendo uno).
    """
    merged = {}
    for h in histories:
        if h is None:
            continue
        for key, vals in h.history.items():
            merged.setdefault(key, []).extend(vals)
    return merged


def plot_history(history_or_dict, model_name, save_path=None):
    """
    Plot de loss y accuracy (train vs val) en dos subplots.

    Acepta tanto un keras.callbacks.History como un dict ya mergeado.
    """
    if hasattr(history_or_dict, "history"):
        h = history_or_dict.history
    else:
        h = history_or_dict

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(h.get("loss", []), label="train")
    if "val_loss" in h:
        axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(h.get("accuracy", []), label="train")
    if "val_accuracy" in h:
        axes[1].plot(h["val_accuracy"], label="val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_and_show(fig, save_path)


# -----------------------------------------------------------------------------
# Evaluación
# -----------------------------------------------------------------------------

def plot_confusion_side_by_side(results_list, class_names, save_path=None):
    """
    Grilla de matrices de confusión normalizadas (una por modelo).

    `results_list`: lista de dicts con claves 'model_name', 'y_true', 'y_pred'.
    """
    n = len(results_list)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, r in zip(axes, results_list):
        cm = confusion_matrix(r["y_true"], r["y_pred"], normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, ax=ax, annot_kws={"size": 8},
        )
        ax.set_title(r["model_name"], fontsize=10)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_tradeoff_scatter(comparison_df, save_path=None):
    """
    Scatter del trade-off F1 macro vs latencia (ms/imagen).

    Tamaño del marcador proporcional a #parámetros. Colorea por familia.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    df = comparison_df.dropna(subset=["ms_por_imagen"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "Sin datos de latencia disponibles",
                ha="center", va="center", transform=ax.transAxes)
        _save_and_show(fig, save_path)
        return

    # Tamaño del marcador escalado a [50, 600]
    params = df["parametros"].fillna(df["parametros"].median())
    if params.max() > params.min():
        sizes = 50 + 550 * (params - params.min()) / (params.max() - params.min())
    else:
        sizes = np.full(len(params), 200)

    families = df["familia"].unique()
    palette = sns.color_palette("tab10", n_colors=len(families))
    color_map = dict(zip(families, palette))
    colors = df["familia"].map(color_map)

    ax.scatter(df["ms_por_imagen"], df["f1_macro"],
               s=sizes, c=list(colors), alpha=0.7, edgecolors="black")

    for _, row in df.iterrows():
        ax.annotate(row["modelo"],
                    (row["ms_por_imagen"], row["f1_macro"]),
                    fontsize=8, alpha=0.85,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Latencia (ms / imagen)")
    ax.set_ylabel("F1 macro")
    ax.set_title("Trade-off: rendimiento vs costo computacional\n"
                 "(tamaño del marcador ∝ #parámetros)")
    ax.grid(alpha=0.3)

    handles = [plt.scatter([], [], color=color_map[f], s=120, edgecolors="black",
                            label=f) for f in families]
    ax.legend(handles=handles, title="Familia", loc="best")

    plt.tight_layout()
    _save_and_show(fig, save_path)


# -----------------------------------------------------------------------------
# Comparación de interpolación (P3, P4)
# -----------------------------------------------------------------------------

def plot_interpolation_comparison(img_original, img_bilinear, img_bicubic,
                                   interp_metrics_df=None, save_path=None):
    """
    Visualización completa de comparación bilinear vs bicubic.

    Row 1: original, bilinear 224, bicubic 224.
    Row 2: |bilinear - bicubic|, FFT log-magnitude de bilinear, tabla de métricas.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1 — imágenes
    axes[0, 0].imshow(img_original.astype("uint8"))
    axes[0, 0].set_title(f"Original {img_original.shape[0]}×{img_original.shape[1]}")
    axes[0, 1].imshow(img_bilinear.astype("uint8"))
    axes[0, 1].set_title(f"Bilinear {img_bilinear.shape[0]}×{img_bilinear.shape[1]}")
    axes[0, 2].imshow(img_bicubic.astype("uint8"))
    axes[0, 2].set_title(f"Bicubic {img_bicubic.shape[0]}×{img_bicubic.shape[1]}")

    # Row 2 — diferencias y FFT
    diff = np.abs(img_bilinear.astype(float) - img_bicubic.astype(float))
    diff_vis = np.clip(diff, 0, 255).astype("uint8")
    axes[1, 0].imshow(diff_vis)
    axes[1, 0].set_title("|Bilinear − Bicubic|")

    gray = img_bilinear.astype(float).mean(axis=-1) if img_bilinear.ndim == 3 else img_bilinear
    fft = np.fft.fftshift(np.fft.fft2(gray))
    log_mag = np.log1p(np.abs(fft))
    axes[1, 1].imshow(log_mag, cmap="magma")
    axes[1, 1].set_title("FFT log-magnitud (bilinear)")

    axes[1, 2].axis("off")
    if interp_metrics_df is not None:
        try:
            table = axes[1, 2].table(
                cellText=interp_metrics_df.round(4).values,
                colLabels=list(interp_metrics_df.columns),
                rowLabels=list(interp_metrics_df.index),
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)
            axes[1, 2].set_title("Métricas empíricas")
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f"Métricas no renderizables:\n{e}",
                            ha="center", va="center", transform=axes[1, 2].transAxes)

    for ax in axes[0]:
        ax.axis("off")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")

    plt.tight_layout()
    _save_and_show(fig, save_path)


def radial_spectrum(image_batch):
    """Espectro radial promedio (magnitud log) sobre el canal de luminancia.
       image_batch: [N, H, W, 3] float32. Devuelve vector de longitud H//2."""
    lum = (0.2126*image_batch[..., 0] + 0.7152*image_batch[..., 1] + 0.0722*image_batch[..., 2])
    F   = np.fft.fftshift(np.fft.fft2(lum, axes=(-2, -1)), axes=(-2, -1))
    mag = np.log1p(np.abs(F))                       # log-magnitud, estabiliza media
    H, W = mag.shape[-2:]
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(np.int32)
    r_max = min(cy, cx)
    # Promedio por anillo radial y luego promedio sobre el batch
    radial = np.zeros((mag.shape[0], r_max), dtype=np.float64)
    for k in range(r_max):
        mask = (r == k)
        radial[:, k] = mag[..., mask].mean(axis=-1)
    return radial.mean(axis=0), r_max