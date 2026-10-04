"""Dataset loading, splitting and tf.data pipelines.

Flow:
    train.csv + train_images/  --load_dataset-->  X (N, 224, 224, 3) uint8,  y (N,) int
    dataframe                  --make_splits-->   train / val / test dataframes (stratified)
    X, y                       --make_tf_dataset--> batches for model.fit / model.predict
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src import config
from src.preprocessing import load_image, preprocess_image


def load_dataset(df, images_dir, img_size=config.IMG_SIZE, use_ben_graham=config.USE_BEN_GRAHAM,
                 workers=8):
    """Read and preprocess every image listed in `df` (columns: id_code, diagnosis).

    APTOS images are large (up to ~3000 px), so they are preprocessed once and kept in
    memory at 224 x 224: 3,662 images x 150 KB ≈ 550 MB.
    Several threads are used because OpenCV releases Python's GIL while decoding.

    Returns: X uint8 (N, img_size, img_size, 3), y int (N,)
    """
    paths = [os.path.join(images_dir, f"{id_code}.png") for id_code in df["id_code"]]

    def process(path):
        return preprocess_image(load_image(path), img_size, use_ben_graham)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(pool.map(process, paths))

    X = np.stack(images).astype(np.uint8)
    y = df["diagnosis"].to_numpy().astype(np.int32)
    return X, y


def make_splits(df, val_fraction=config.VAL_FRACTION, test_fraction=config.TEST_FRACTION,
                seed=config.SEED):
    """Stratified train / validation / test split (70 / 15 / 15 by default).

    "Stratified" keeps the class proportions equal in all three parts, which matters
    because APTOS is imbalanced (about half the images are class 0).
    Returns: (train_df, val_df, test_df)
    """
    # exact image counts (not fractions) avoid floating-point rounding surprises
    n_test = round(len(df) * test_fraction)
    n_val = round(len(df) * val_fraction)
    train_val_df, test_df = train_test_split(
        df, test_size=n_test, stratify=df["diagnosis"], random_state=seed)
    train_df, val_df = train_test_split(
        train_val_df, test_size=n_val, stratify=train_val_df["diagnosis"], random_state=seed)
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def load_or_create_splits(df, splits_dir):
    """Use shared split files if they exist (team), otherwise create and save them.

    Files: train_split.csv, val_split.csv, test_split.csv with columns id_code, diagnosis.
    """
    names = ["train_split.csv", "val_split.csv", "test_split.csv"]
    paths = [os.path.join(splits_dir, n) for n in names]
    if all(os.path.exists(p) for p in paths):
        print(f"Using existing split files from {splits_dir}")
        return tuple(pd.read_csv(p) for p in paths)

    splits = make_splits(df)
    os.makedirs(splits_dir, exist_ok=True)
    for split_df, path in zip(splits, paths):
        split_df[["id_code", "diagnosis"]].to_csv(path, index=False)
    print(f"Created new split files in {splits_dir}")
    return splits


def build_augmenter(seed=config.SEED):
    """Random changes applied to training images only (settings doc, section 5).

    Each epoch the model sees slightly different versions of the same retina,
    which reduces overfitting on a small dataset.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        keras.layers.RandomRotation(20 / 360, fill_mode="constant", seed=seed),   # ±20°
        keras.layers.RandomZoom(0.1, fill_mode="constant", seed=seed),            # ±10%
        keras.layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),      # ±10%
    ], name="augmenter")


def make_tf_dataset(X, y, batch_size=config.BATCH_SIZE, training=False, seed=config.SEED):
    """Wrap arrays in a tf.data pipeline.

    training=True  -> shuffle + augmentation (train split)
    training=False -> fixed order, no augmentation (val / test splits)
    Images stay uint8 in memory and are converted to float32 per batch.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda images, labels: (tf.cast(images, tf.float32), labels),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda images, labels: (augmenter(images, training=True), labels),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
