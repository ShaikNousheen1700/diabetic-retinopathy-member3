"""End-to-end smoke test: split -> preprocess -> train (both phases) -> save -> evaluate -> predict.

Uses 50 synthetic images, random weights and 1 epoch per phase, so it checks that the
whole pipeline runs and produces the right files — not model accuracy.
"""

import json
import os

import cv2
import pandas as pd

from src import config
from src.data import make_splits
from src.evaluate import evaluate_model, save_results
from src.predict import DRPredictor
from src.train import get_class_weights, run


def test_splits_are_stratified_and_disjoint():
    df = pd.DataFrame({"id_code": [f"i{i}" for i in range(200)], "diagnosis": [i % 5 for i in range(200)]})
    train, val, test = make_splits(df)
    assert (len(train), len(val), len(test)) == (140, 30, 30)
    ids = [set(s["id_code"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for split in (train, val, test):
        assert split["diagnosis"].value_counts().nunique() == 1    # equal classes stay equal


def test_class_weights_favour_rare_classes():
    y = [0] * 80 + [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2
    w = get_class_weights(y)
    assert w[4] > w[3] > w[2] > w[1] > w[0]


def test_full_pipeline(fake_aptos, tmp_path):
    out_dir = tmp_path / "output"
    result = run(str(fake_aptos), str(out_dir), weights=None,
                 phase1_epochs=1, phase2_epochs=1, batch_size=8)

    # files written by training
    model_path = out_dir / config.MODEL_FILENAME
    assert model_path.exists()
    assert (out_dir / config.MODEL_INFO_FILENAME).exists()
    for name in ("train_split.csv", "val_split.csv", "test_split.csv"):
        assert (out_dir / "splits" / name).exists()
    assert len(result["history"]["loss"]) == 2               # 1 epoch per phase

    # evaluation
    metrics, y_pred, probs = evaluate_model(result["model"], result["X_test"], result["y_test"],
                                            str(model_path))
    for key in ("accuracy", "macro_f1", "quadratic_weighted_kappa", "confusion_matrix"):
        assert key in metrics
    paths = save_results(metrics, result["history"], str(tmp_path / "results"))
    assert all(os.path.exists(p) for p in paths)
    json.load(open(paths[0]))

    # sample prediction from the saved file (same path the backend will use)
    predictor = DRPredictor(str(model_path))
    sample = fake_aptos / "train_images" / "img000.png"
    pred = predictor.predict_path(str(sample))
    assert pred["class_name"] in config.CLASS_NAMES
    assert abs(sum(pred["probabilities"].values()) - 1) < 1e-3

    # bytes input (how the API receives uploads) gives the same answer
    pred_bytes = predictor.predict_bytes(sample.read_bytes())
    assert pred_bytes == pred
