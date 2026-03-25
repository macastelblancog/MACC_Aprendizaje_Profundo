# Esto es para construir los modelos
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Conv2D,
    MaxPooling2D, GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import json
with open('./src/utils/convolutional.json', 'r') as file:
    detault_Attributes = json.load(file)

def balanced_accuracy(y_true, y_pred):
    """
    Mean per-class recall (balanced accuracy) — graph-safe.
    Penalizes models that exploit class imbalance.
    """
    n_classes = tf.shape(y_pred)[1]
    y_true    = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred    = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)

    # Build confusion matrix as a flat one-hot sum
    conf = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes)
    conf = tf.cast(conf, tf.float32)

    # Row sums = number of true samples per class
    row_sums = tf.reduce_sum(conf, axis=1)

    # Diagonal = true positives per class → recall per class
    recalls = tf.linalg.diag_part(conf) / (row_sums + K.epsilon())

    return tf.reduce_mean(recalls)


def augment_minority_classes(X, Y, key="train", n_minority = 3, flips = ("horizontal", "vertical", "both")):
    """
    Augments the n_minority least-represented classes by flipping images.
    Assumes squared input images (28x28xC).
    
    Flips applied per minority class:
        "horizontal" → left-right flip  (axis=1)
        "vertical"   → up-down flip     (axis=0)
        "both"       → 180° rotation    (axis=0 and axis=1)
    
    Returns new X and Y dicts with augmented samples appended.
    """
    y = np.ravel(Y[key])
    X_split = X[key]                          # (N, 28, 28, C)

    # Identify the n_minority least-represented classes
    classes, counts = np.unique(y, return_counts=True)
    minority_classes = classes[np.argsort(counts)[:n_minority]]
    print(f"Augmenting classes: {minority_classes} "
          f"(counts: {counts[np.argsort(counts)[:n_minority]]})")

    augmented_X = []
    augmented_Y = []

    for cls, flip_type in zip(minority_classes, flips):
        # Boolean mask → index filter
        mask    = y == cls
        X_cls   = X_split[mask]               # (n_cls, 28, 28, C)

        if flip_type == "horizontal":
            X_flipped = X_cls[:, :, ::-1, :]  # flip width axis
        elif flip_type == "vertical":
            X_flipped = X_cls[:, ::-1, :, :]  # flip height axis
        elif flip_type == "both":
            X_flipped = X_cls[:, ::-1, ::-1, :]  # 180° rotation

        augmented_X.append(X_flipped)
        augmented_Y.append(np.full(X_flipped.shape[0], cls))

    # Stack all augmented samples and append to original split
    X_aug = np.concatenate(augmented_X, axis = 0)
    Y_aug = np.concatenate(augmented_Y, axis = 0)

    X_new = np.concatenate([X_split, X_aug], axis = 0)
    Y_new = np.concatenate([y,        Y_aug], axis = 0)

    # Shuffle to avoid the model seeing augmented blocks last
    idx     = np.random.permutation(len(X_new))
    X_new   = X_new[idx]
    Y_new   = Y_new[idx].reshape(-1, 1)

    # Return new dicts preserving all other splits untouched
    X_out   = {**X, key: X_new}
    Y_out   = {**Y, key: Y_new}

    return X_out, Y_out


def compute_class_weights(Y, key="train"):
    y = np.ravel(Y[key])
    
    classes, counts = np.unique(y, return_counts=True)
    
    total = len(y)
    n_classes = len(classes)
    
    class_weights = {
        int(cls): total / (n_classes * count)
        for cls, count in zip(classes, counts)
    }
    
    return class_weights


class skin_classifier:
    def __init__(
        self
        ,X : dict ,Y : dict
        ,attributes : dict =  detault_Attributes
    ):
        """
        CNN classifier configurable mediante diccionario de atributos.

        Parámetros esperados en attributes:
        {
            "conv_layers": [
                {
                    "filters": 16,
                    "kernel_size": (3, 3),
                    "activation": "relu",
                    "padding": "same",
                    "strides": (1, 1),
                    "pool_size": (2, 2),
                    "dropout": 0.2
                }
            ],
            "dense_layers": [
                {
                    "units": 32,
                    "activation": "relu",
                    "dropout": 0.2
                }
            ]
        }
        """
        self.X = X
        self.Y = Y
        self.attributes = attributes
        self.history = None
        self.model = self.__build_layers()

        try:
            plot_model(self.model, show_shapes=True)
        except Exception as e:
            print(f"Could not render plot_model: {e}")

    def __build_layers(self):
        input_shape = self.X["train"].shape[1:]
        n_classes = len(np.unique(np.ravel(self.Y["train"])))

        layers = [Input(shape=input_shape, name = "input")]

        # Bloques convolucionales
        for index, conv_cfg in enumerate(self.attributes.get("conv_layers", []), start=1):
            layers.append(
                Conv2D(
                    filters = conv_cfg["filters"]
                    ,kernel_size = conv_cfg.get("kernel_size", (3, 3))
                    ,activation = conv_cfg.get("activation", "relu")
                    ,padding = conv_cfg.get("padding", "same")
                    ,strides = conv_cfg.get("strides", (1, 1))
                    ,name = f"conv{index}"
                )
            )

            layers.append(
                BatchNormalization(name=f"bn{index}")  # ← agregar aquí
            )

            pool_size = conv_cfg.get("pool_size", None)
            if pool_size is not None:
                layers.append(
                    MaxPooling2D(
                        pool_size = pool_size
                        ,name = f"pool{index}"
                    )
                )

            dropout_rate = conv_cfg.get("dropout", 0.0)
            if dropout_rate > 0:
                layers.append(
                    Dropout(dropout_rate, name = f"conv_dropout{index}")
                )

        # Reducción espacial para clasificación
        layers.append(GlobalAveragePooling2D(name = "global_avg_pool"))

        # Capas densas
        for index, dense_cfg in enumerate(self.attributes.get("dense_layers", []), start = 1):
            layers.append(
                Dense(
                    units = dense_cfg["units"]
                    ,activation=dense_cfg.get("activation", "relu")
                    ,name = f"hidden{index}"
                )
            )

            dropout_rate = dense_cfg.get("dropout", 0.0)
            if dropout_rate > 0:
                layers.append(
                    Dropout(dropout_rate, name=f"dense_dropout{index}")
                )

        # Salida
        layers.append(
            Dense(
                units = n_classes,
                activation = "softmax",
                name = "output"
            )
        )

        return Sequential(layers=layers)


    def fit(
        self,
        X,
        Y,
        train_tag,
        validation_tag,
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        use_live_loss=False
    ):
        self.model.compile(
            optimizer = Adam(learning_rate=learning_rate),
            loss = SparseCategoricalCrossentropy(),
            metrics = [balanced_accuracy]
        )

        callbacks = [
            EarlyStopping(
                monitor              = "val_balanced_accuracy"
                ,patience             = 15
                ,restore_best_weights = True
                ,mode                 = "max"
                ,min_delta            = 0.001  # ignore improvements smaller than 0.1%
                ,start_from_epoch     = 40  
    
            ),
            ModelCheckpoint(                            # ← agregar aquí
                filepath        = "./models/dermClass.keras",
                monitor         = "val_balanced_accuracy",
                save_best_only  = True,
                mode            = "max",
                verbose         = 1
            )
        ]

        if use_live_loss:
            try:
                from livelossplot import PlotLossesKeras
                callbacks.append(PlotLossesKeras())
            except Exception as e:
                print(f"PlotLossesKeras could not be loaded: {e}")

        y_train = np.ravel(Y[train_tag])
        y_val = np.ravel(Y[validation_tag])

        class_weights = compute_class_weights(Y, key="train")

        self.history = self.model.fit(
            X[train_tag]
            ,y_train
            ,epochs = epochs
            ,batch_size = batch_size
            ,validation_data = (X[validation_tag], y_val)
            ,class_weight = class_weights 
            ,callbacks = callbacks
        )

        return self.history

    def metrics_report(self, X, Y):
        matrixes = {}

        for label in X.keys():
            y_true = np.ravel(Y[label])
            y_pred = np.argmax(self.model.predict(X[label], verbose=0), axis=1)

            # Confusion matrix normalized by true labels (row-wise)
            conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
            conf_matrix_df = pd.DataFrame(
                conf_matrix,
                index   = [f"True {i}" for i in range(conf_matrix.shape[0])],
                columns = [f"Pred {i}" for i in range(conf_matrix.shape[1])]
            ).round(3)
            matrixes[label] = conf_matrix_df

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                conf_matrix_df,
                annot  = True,
                fmt    = ".1%",       # shows 0.873 as 87.3%
                cmap   = "Blues",
                vmin   = 0,
                vmax   = 1,           # fixes the color scale across splits
                cbar   = True,
                ax     = ax
            )
            ax.set_title(f"Confusion Matrix (normalized by true label) — {label}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            plt.savefig(f"./figures/conf_matrix_{label}.png", dpi=150, bbox_inches = "tight")
            plt.show()

            # Classification report — weighted avg is now meaningful
            report = classification_report(
                y_true,
                y_pred,
                output_dict  = True,
                zero_division = np.nan
            )
            report_df = pd.DataFrame(report).T.round(3)

            print(label)
            #display(report_df)

        return matrixes