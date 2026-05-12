"""
Pipeline de preprocesamiento por familia de modelo.

Construye `tf.data.Dataset` con:
- Interpolación configurable (bilinear / bicubic) para upsampling 28→224.
- Normalización específica por familia (baseline, mobilenet, vit, hybrid_features).
- Data augmentation opcional (importado desde `src/augmentation.py`).
"""
import tensorflow as tf

from src.config import TARGET_SIZE, BATCH_SIZE, SEED
from src.augmentation import augmenter_224, augmenter_28


AUTOTUNE = tf.data.AUTOTUNE

VALID_FAMILIES = ("baseline", "mobilenet", "vit", "hybrid_features")


def resize_tf(images, size, method):
    """Resize con cast a float32 antes de interpolar."""
    return tf.image.resize(
        tf.cast(images, tf.float32), (size, size), method=method
    )


def make_dataset(images, labels, batch_size=BATCH_SIZE,
                 shuffle=False, model_family="baseline",
                 interpolation="bilinear", augment=False):
    """
    Construye un tf.data.Dataset listo para entrenamiento o evaluación.

    Parameters
    ----------
    images : np.ndarray
        Tensor (N, 28, 28, 3) uint8.
    labels : np.ndarray
        Etiquetas (N,) int32.
    batch_size : int
    shuffle : bool
        Si True, baraja con seed reproducible por epoch.
    model_family : str
        Una de {'baseline', 'mobilenet', 'vit', 'hybrid_features'}.
    interpolation : str
        Método de upsampling: 'bilinear' o 'bicubic'.
    augment : bool
        Si True, aplica el augmenter correspondiente a la resolución.

    Returns
    -------
    tf.data.Dataset con (x, y) batched y prefetched.
    """
    if model_family not in VALID_FAMILIES:
        raise ValueError(
            f"model_family desconocido: {model_family!r}. "
            f"Válidos: {VALID_FAMILIES}"
        )

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(labels),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _map_baseline(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        if augment:
            x = augmenter_28(x, training=True)
        return x, y

    def _map_mobilenet(x, y):
        x = resize_tf(x, TARGET_SIZE, interpolation)
        if augment:
            x = augmenter_224(x, training=True)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    def _map_vit(x, y):
        x = resize_tf(x, TARGET_SIZE, interpolation)
        if augment:
            x = augmenter_224(x, training=True)
        x = tf.cast(x, tf.float32)  # ViT maneja normalización internamente
        return x, y

    fn_map = {
        "baseline": _map_baseline,
        "mobilenet": _map_mobilenet,
        "hybrid_features": _map_mobilenet,  # mismo preproceso que mobilenet
        "vit": _map_vit,
    }

    ds = ds.map(fn_map[model_family], num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds