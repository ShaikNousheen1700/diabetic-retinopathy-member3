"""Two-phase DenseNet121 training.

Phase 1 — base frozen, train only the new classifier head (lr 1e-3).
          The head starts random; training it first stops large random gradients
          from damaging the pretrained DenseNet weights.
Phase 2 — unfreeze the last dense block and fine-tune with a small lr (1e-5),
          with early stopping on validation loss.

Can be run from Colab (import the functions) or the command line:
    python -m src.train --data-dir /content/aptos --out-dir /content/output
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src import config
from src.data import load_dataset, load_or_create_splits, make_tf_dataset
from src.model import build_densenet121, compile_model, unfreeze_top


def get_class_weights(y_train, num_classes=config.NUM_CLASSES):
    """'Balanced' class weights: weight_c = N / (num_classes * count_c).

    Rare classes (e.g. Severe) get a larger weight, so mistakes on them cost more in the loss.
    Returns: {class_id: weight}
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_two_phase(model, base, train_ds, val_ds, class_weight, out_dir,
                    phase1_epochs=config.PHASE1_EPOCHS, phase2_epochs=config.PHASE2_EPOCHS):
    """Run phase 1 and phase 2. The best model (lowest val_loss) is saved to out_dir.

    Returns: history dict with lists per metric, plus 'phase1_epochs' (where phase 2 starts).
    """
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, config.MODEL_FILENAME)
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)

    # ---- Phase 1: head only ----
    print(f"\n=== Phase 1: training head only ({phase1_epochs} epochs, lr={config.PHASE1_LR}) ===")
    compile_model(model, config.PHASE1_LR)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=phase1_epochs,
                   class_weight=class_weight, callbacks=[checkpoint], verbose=2)

    # ---- Phase 2: fine-tune last dense block ----
    n_trainable = unfreeze_top(base)
    print(f"\n=== Phase 2: fine-tuning {n_trainable} base layers "
          f"(up to {phase2_epochs} epochs, lr={config.PHASE2_LR}) ===")
    compile_model(model, config.PHASE2_LR)        # must re-compile after changing `trainable`
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.EARLY_STOP_PATIENCE, restore_best_weights=True)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=phase2_epochs,
                   class_weight=class_weight, callbacks=[checkpoint, early_stop], verbose=2)

    history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
    history["phase1_epochs"] = len(h1.history["loss"])
    return history


def run(data_dir, out_dir, splits_dir=None, weights="imagenet", limit=None,
        phase1_epochs=config.PHASE1_EPOCHS, phase2_epochs=config.PHASE2_EPOCHS,
        batch_size=config.BATCH_SIZE):
    """End-to-end: read data -> split -> preprocess -> train -> save model + history.

    data_dir:   folder with train.csv and train_images/
    out_dir:    where the .keras model, history and info JSON are written
    splits_dir: where split CSVs are read from / written to (default: out_dir/splits)
    limit:      use only the first N images (quick testing)
    Returns: dict with model, history and the split arrays (used by the evaluation step).
    """
    keras.utils.set_random_seed(config.SEED)      # same seed for Python, NumPy and TensorFlow
    splits_dir = splits_dir or os.path.join(out_dir, "splits")

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    if limit:
        df = df.head(limit)
    train_df, val_df, test_df = load_or_create_splits(df, splits_dir)
    print(f"Split sizes: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    images_dir = os.path.join(data_dir, "train_images")
    t0 = time.time()
    X_train, y_train = load_dataset(train_df, images_dir)
    X_val, y_val = load_dataset(val_df, images_dir)
    X_test, y_test = load_dataset(test_df, images_dir)
    print(f"Preprocessed {len(X_train) + len(X_val) + len(X_test)} images in {time.time() - t0:.0f}s")

    class_weight = get_class_weights(y_train)
    print("Class weights:", {k: round(v, 2) for k, v in class_weight.items()})

    train_ds = make_tf_dataset(X_train, y_train, batch_size, training=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size)

    model, base = build_densenet121(weights=weights)
    t0 = time.time()
    history = train_two_phase(model, base, train_ds, val_ds, class_weight, out_dir,
                              phase1_epochs, phase2_epochs)
    train_seconds = time.time() - t0
    history["train_seconds"] = train_seconds
    print(f"Training finished in {train_seconds / 60:.1f} min")

    with open(os.path.join(out_dir, f"{config.MODEL_NAME}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Small info file stored next to the model — the backend reads it for GET /model-info
    info = {
        "model_name": "DenseNet121",
        "model_file": config.MODEL_FILENAME,
        "framework": f"TensorFlow {tf.__version__} / Keras {keras.__version__}",
        "input_size": [config.IMG_SIZE, config.IMG_SIZE, 3],
        "class_names": config.CLASS_NAMES,
        "preprocessing": {
            "steps": ["BGR->RGB", "crop black border", f"resize {config.IMG_SIZE}x{config.IMG_SIZE}",
                      "ImageNet normalisation (inside model)"],
            "use_ben_graham": config.USE_BEN_GRAHAM,
        },
        "weights": weights or "random",
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "epochs_trained": len(history["loss"]),
        "train_seconds": round(train_seconds, 1),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, config.MODEL_INFO_FILENAME), "w") as f:
        json.dump(info, f, indent=2)

    best_model = keras.models.load_model(os.path.join(out_dir, config.MODEL_FILENAME))
    return {"model": best_model, "history": history, "info": info,
            "X_test": X_test, "y_test": y_test, "test_df": test_df}


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 on APTOS 2019")
    parser.add_argument("--data-dir", required=True, help="folder with train.csv and train_images/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument("--phase1-epochs", type=int, default=config.PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=config.PHASE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    run(args.data_dir, args.out_dir, args.splits_dir,
        weights=None if args.weights == "none" else args.weights, limit=args.limit,
        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size)


if __name__ == "__main__":
    main()
