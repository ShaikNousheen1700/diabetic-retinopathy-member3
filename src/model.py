"""DenseNet121 model for 5-class Diabetic Retinopathy classification.

Architecture (data flows top to bottom):

    Input (224, 224, 3)  RGB, values 0-255
    Rescaling(1/255)     -> values 0-1
    Normalization        -> subtract ImageNet mean, divide by ImageNet std
    DenseNet121 base     -> (7, 7, 1024) feature maps   [ImageNet-pretrained]
    GlobalAveragePooling -> (1024,)  one number per feature map
    Dropout(0.3)         -> randomly zeroes 30% of values during training only
    Dense(5, softmax)    -> (5,)  probability for each DR class, sums to 1

Putting the normalisation inside the model means the saved .keras file
contains everything needed: the backend only has to crop + resize the image.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config


def build_densenet121(img_size=config.IMG_SIZE, num_classes=config.NUM_CLASSES,
                      dropout=config.DROPOUT, weights="imagenet"):
    """Build the full model with the DenseNet121 base frozen (phase 1 setup).

    weights: "imagenet" for transfer learning, or None for random weights (used in tests).
    Returns: (model, base) — `base` is the DenseNet121 part, needed later for fine-tuning.
    """
    base = keras.applications.DenseNet121(
        include_top=False,                      # drop the 1000-class ImageNet classifier
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False                      # phase 1: only the new head learns

    mean = np.array(config.IMAGENET_MEAN, dtype="float32")
    std = np.array(config.IMAGENET_STD, dtype="float32")

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    x = keras.layers.Normalization(mean=mean, variance=std ** 2, name="imagenet_norm")(x)
    # training=False keeps BatchNorm layers in inference mode, even when fine-tuning.
    # This is the recommended setting for transfer learning on small datasets.
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="densenet121_dr")
    return model, base


def unfreeze_top(base, from_layer=config.FINE_TUNE_FROM):
    """Phase 2 setup: make the last dense block trainable, keep everything before it frozen.

    BatchNormalization layers stay frozen: with small batches their statistics are noisy.
    Returns the number of trainable layers in the base.
    """
    base.trainable = True
    names = [layer.name for layer in base.layers]
    start = names.index(from_layer)
    for i, layer in enumerate(base.layers):
        is_bn = isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = i >= start and not is_bn
    return sum(layer.trainable for layer in base.layers)


def compile_model(model, learning_rate):
    """Attach optimizer, loss and metric.

    sparse_categorical_crossentropy: labels are plain integers 0-4 (not one-hot vectors).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
