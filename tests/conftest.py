"""Shared test fixtures: synthetic fundus-like images and a tiny fake APTOS dataset.

The real dataset lives in Google Drive (Colab only), so tests use generated images:
a coloured circle (the "retina") on a black background, like a real fundus photo.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

# make `import src...` work when running pytest from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_fundus(height=300, width=400, seed=0):
    """RGB uint8 image: orange-red circle on black, with a few dark 'lesion' dots."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(height, width) // 2 - 10
    cv2.circle(img, center, radius, (200, 90, 40), thickness=-1)
    for _ in range(5):
        x, y = rng.integers(center[0] - radius // 2, center[0] + radius // 2, size=2)
        cv2.circle(img, (int(x), int(y)), 4, (120, 30, 20), thickness=-1)
    return img


@pytest.fixture
def fundus_rgb():
    return make_fundus()


@pytest.fixture
def fake_aptos(tmp_path):
    """Folder with train.csv + train_images/ (50 PNGs, 10 per class), same layout as APTOS."""
    images_dir = tmp_path / "train_images"
    images_dir.mkdir()
    rows = []
    for i in range(50):
        id_code = f"img{i:03d}"
        label = i % 5
        img = make_fundus(280 + i, 360 + i, seed=i)
        cv2.imwrite(str(images_dir / f"{id_code}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({"id_code": id_code, "diagnosis": label})
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    return tmp_path
