"""Carga DermaMNIST y aplica subsetting estratificado para FAST_MODE."""
import numpy as np
import medmnist
from medmnist import INFO
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import DATA_FLAG, SOURCE_SIZE, SEED


def load_dermamnist():
    """
    Descarga (si es necesario) y carga DermaMNIST en sus tres splits.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test : np.ndarray
        Imágenes RGB (N, 28, 28, 3) uint8 y etiquetas int32.
    class_names : list[str]
        Nombres de las 7 clases ordenados por índice.
    mel_idx : int
        Índice de la clase Melanoma (clase crítica).
    """
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info["python_class"])

    splits = {}
    for split in ["train", "val", "test"]:
        ds = DataClass(split=split, download=True, size=SOURCE_SIZE, as_rgb=True)
        splits[split] = (
            ds.imgs.copy(),
            ds.labels.squeeze().astype("int32"),
        )

    label_map = {int(k): v for k, v in info["label"].items()}
    class_names = [label_map[i] for i in range(len(label_map))]
    mel_idx = next(i for i, name in label_map.items() if "mel" in name.lower())

    return (
        splits["train"][0], splits["train"][1],
        splits["val"][0],   splits["val"][1],
        splits["test"][0],  splits["test"][1],
        class_names, mel_idx,
    )


def stratified_subset(x, y, max_samples, seed=SEED):
    """
    Submuestreo estratificado preservando la distribución de clases.

    Si `max_samples` es None o mayor que len(y), retorna (x, y) sin cambios.
    """
    if max_samples is None or len(y) <= max_samples:
        return x, y

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=max_samples, random_state=seed
    )
    idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return x[idx], y[idx]