"""
Utilidades de entrenamiento: class weights, callbacks y fit con timer.

- get_class_weights(): pesos balanceados para mitigar desbalance de clases
  (clave para mejorar mel_recall).
- get_callbacks(): EarlyStopping + ReduceLROnPlateau + ModelCheckpoint opcional.
- timed_fit(): wrapper de model.fit() que mide tiempo de entrenamiento
  (FIX MEJORA-02).
"""
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from src.config import NUM_CLASSES


def get_class_weights(y_train):
    """
    Calcula class_weight balanceado para 7 clases.

    Returns
    -------
    dict[int, float] mapeando índice de clase -> peso.
    """
    values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    return {i: float(w) for i, w in enumerate(values)}


def get_callbacks(monitor="val_loss",
                  patience_es=4,
                  patience_lr=2,
                  checkpoint_path=None,
                  min_lr=1e-6):
    """
    Construye la lista estándar de callbacks de entrenamiento.

    Parameters
    ----------
    monitor : str
        Métrica a monitorear ('val_loss' o 'val_accuracy').
    patience_es : int
        Paciencia de EarlyStopping (epochs sin mejora).
    patience_lr : int
        Paciencia de ReduceLROnPlateau.
    checkpoint_path : str | Path | None
        Si se provee, agrega ModelCheckpoint guardando el mejor modelo.
    min_lr : float
        Learning rate mínimo para ReduceLROnPlateau.

    Returns
    -------
    list[keras.callbacks.Callback]
    """
    cbs = [
        EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=patience_lr,
            min_lr=min_lr,
        ),
    ]
    if checkpoint_path is not None:
        cbs.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
            )
        )
    return cbs


def timed_fit(model, train_ds, val_ds, epochs,
              class_weights=None, callbacks=None, verbose=1):
    """
    Wrapper de model.fit() que mide el tiempo de pared transcurrido.

    Returns
    -------
    history : keras.callbacks.History
    elapsed_seconds : float
        Tiempo total de entrenamiento en segundos (incluyendo validación).
    """
    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    return history, elapsed