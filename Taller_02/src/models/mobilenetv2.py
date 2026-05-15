"""
Builders para MobileNetV2: Feature Extraction (FE) y Fine-Tuning (FT).

Incluye:
- build_mobilenet_fe(): base congelada (T1).
- unfreeze_for_finetuning(): descongela últimas N capas (FIX BUG-09, T2).
- snapshot_model(): copia el modelo FE antes de modificar trainable (FIX BUG-02).
- build_embedding_extractor(): extrae features hasta GAP para Hybrid RF (T5).
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy


from src.config import NUM_CLASSES, LR_FE, LR_FT
from src.training import balanced_accuracy


def build_mobilenet_fe(num_classes=NUM_CLASSES,
                       input_shape=(224, 224, 3),
                       lr=LR_FE,
                       dropout=0.3):
    """
    Feature Extraction: MobileNetV2 ImageNet con base completamente congelada.

    Returns
    -------
    model : keras.Model compilado
    base  : keras.Model (la base MobileNetV2, necesaria para FT posterior)
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # congelar TODO

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_feature_extraction")
    model.compile(
    optimizer=Adam(learning_rate=lr),
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy", balanced_accuracy],
)
    return model, base


def unfreeze_for_finetuning(model, base, n_unfreeze=30, lr=LR_FT):
    """
    Descongela las últimas `n_unfreeze` capas de la base para fine-tuning.

    CRÍTICO (FIX BUG-09): no descongelar la base completa. Con lr muy bajo,
    descongelar los últimos ~30 layers de MobileNetV2 evita destruir las
    representaciones pre-entrenadas y reduce el riesgo de overfitting.

    Re-compila el modelo con el nuevo lr; debe llamarse ANTES de fit().
    """
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in base.layers[-n_unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

@tf.keras.utils.register_keras_serializable(package="src")
def snapshot_model(model, lr=LR_FE):
    """
    Copia profunda del modelo con sus pesos actuales (FIX BUG-02).
    Usa custom_object_scope para que clone_model resuelva balanced_accuracy
    durante la deserialización de la config compilada.
    """
    from src.training import balanced_accuracy   # import local evita circulares

    with keras.saving.custom_object_scope({"balanced_accuracy": balanced_accuracy}):
        snapshot = keras.models.clone_model(model)

    snapshot.set_weights(model.get_weights())
    snapshot.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", balanced_accuracy],  # mantener métrica en el snapshot
    )
    return snapshot

def build_embedding_extractor(model, layer_name="gap"):
    """
    Crea un sub-modelo que retorna los embeddings de `layer_name`.

    Usado por el pipeline Hybrid CNN + Random Forest (T5).
    """
    return keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output,
        name="mobilenet_embedding_extractor",
    )