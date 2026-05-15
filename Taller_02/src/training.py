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

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from src.config import NUM_CLASSES


import tensorflow as tf
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable(package="src")
def balanced_accuracy(y_true, y_pred):
    """Mean per-class recall — graph-safe. Portada de Taller 01."""
    n_classes = tf.shape(y_pred)[1]
    y_true    = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred    = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    conf      = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes)
    conf      = tf.cast(conf, tf.float32)
    row_sums  = tf.reduce_sum(conf, axis=1)
    recalls   = tf.linalg.diag_part(conf) / (row_sums + K.epsilon())
    return tf.reduce_mean(recalls)

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
                  min_lr=1e-6,
                  min_delta=0.001,
                  start_from_epoch=0):
    """
    Parameters agregados vs versión anterior:
      mode            — inferido del monitor (max para accuracy, min para loss)
      min_delta       — ignora mejoras menores (igual que T1)
      start_from_epoch— permite absorber spikes iniciales (crítico para FT)
    """
    mode = "max" if "accuracy" in monitor else "min"

    cbs = [
        EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
            mode=mode,
            min_delta=min_delta,
            start_from_epoch=start_from_epoch,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=patience_lr,
            min_lr=min_lr,
            mode=mode,
        ),
    ]
    if checkpoint_path is not None:
        cbs.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode=mode,
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

