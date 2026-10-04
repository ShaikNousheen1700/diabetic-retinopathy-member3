"""Evaluation of the trained model on the test split.

Produces the metrics required by the settings doc (section 8):
accuracy, macro F1, Quadratic Weighted Kappa, per-class report, confusion matrix,
training curves, parameter count, model file size and inference time per image.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed (works on servers/Colab)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)

from src import config


def evaluate_model(model, X_test, y_test, model_path=None, batch_size=config.BATCH_SIZE):
    """Predict the test split and compute all metrics.

    Returns: (metrics dict, y_pred array, probabilities array (N, 5))
    """
    t0 = time.time()
    probs = model.predict(X_test.astype("float32"), batch_size=batch_size, verbose=0)
    seconds = time.time() - t0
    y_pred = probs.argmax(axis=1)

    labels = list(range(config.NUM_CLASSES))
    metrics = {
        "model": "DenseNet121",
        "test_images": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0)),
        # QWK: agreement between true and predicted grades; predicting 4 for a true 0
        # is penalised much more than predicting 1. Official APTOS metric. 1.0 = perfect.
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_test, y_pred, weights="quadratic")),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, target_names=config.CLASS_NAMES,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "total_params": int(model.count_params()),
        "inference_ms_per_image": round(1000 * seconds / max(len(y_test), 1), 2),
    }
    if model_path and os.path.exists(model_path):
        metrics["model_file_mb"] = round(os.path.getsize(model_path) / 1e6, 1)
    return metrics, y_pred, probs


def plot_confusion_matrix(cm, path):
    """Heatmap: rows = true class, columns = predicted class. Diagonal = correct."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DenseNet121 — Confusion matrix (test split)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curves(history, path):
    """Accuracy and loss per epoch for train and validation; dashed line = start of fine-tuning."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ["accuracy", "loss"]):
        ax.plot(epochs, history[key], label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], label=f"val {key}")
        if "phase1_epochs" in history:
            ax.axvline(history["phase1_epochs"] + 0.5, color="gray", linestyle="--",
                       label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_title(f"DenseNet121 — {key}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_results(metrics, history, results_dir):
    """Write metrics JSON and both plots. Returns the list of file paths written."""
    os.makedirs(results_dir, exist_ok=True)
    name = config.MODEL_NAME
    paths = [os.path.join(results_dir, f"{name}_metrics.json"),
             os.path.join(results_dir, f"{name}_confusion_matrix.png"),
             os.path.join(results_dir, f"{name}_training_curves.png")]
    with open(paths[0], "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(metrics["confusion_matrix"], paths[1])
    plot_training_curves(history, paths[2])
    return paths


def print_summary(metrics):
    print(f"Test images            : {metrics['test_images']}")
    print(f"Accuracy               : {metrics['accuracy']:.4f}")
    print(f"Macro F1               : {metrics['macro_f1']:.4f}")
    print(f"Quadratic Weighted Kappa: {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"Inference time / image : {metrics['inference_ms_per_image']} ms")
    print("\nPer-class recall:")
    for name in config.CLASS_NAMES:
        r = metrics["classification_report"][name]
        print(f"  {name:<17} recall={r['recall']:.3f}  precision={r['precision']:.3f}  n={int(r['support'])}")
