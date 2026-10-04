import cv2
import numpy as np
import pytest

from src.preprocessing import (ben_graham, crop_black_border, decode_image_bytes, load_image,
                               preprocess_image)


def test_crop_removes_black_border(fundus_rgb):
    cropped = crop_black_border(fundus_rgb)
    # the 300x400 image has a circle of radius 140 -> about 281x281 after cropping
    assert cropped.shape[0] < fundus_rgb.shape[0]
    assert cropped.shape[1] < fundus_rgb.shape[1]
    assert abs(cropped.shape[0] - cropped.shape[1]) <= 2


def test_crop_all_black_image_unchanged():
    black = np.zeros((50, 60, 3), dtype=np.uint8)
    assert crop_black_border(black).shape == black.shape


def test_preprocess_output_shape_and_type(fundus_rgb):
    out = preprocess_image(fundus_rgb)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_ben_graham_keeps_shape(fundus_rgb):
    out = ben_graham(fundus_rgb)
    assert out.shape == fundus_rgb.shape and out.dtype == np.uint8


def test_load_image_returns_rgb(tmp_path, fundus_rgb):
    path = tmp_path / "x.png"
    cv2.imwrite(str(path), cv2.cvtColor(fundus_rgb, cv2.COLOR_RGB2BGR))
    np.testing.assert_array_equal(load_image(path), fundus_rgb)   # PNG is lossless


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_image(tmp_path / "missing.png")


def test_decode_bytes_valid_and_corrupted(fundus_rgb):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(fundus_rgb, cv2.COLOR_RGB2BGR))
    np.testing.assert_array_equal(decode_image_bytes(encoded.tobytes()), fundus_rgb)
    with pytest.raises(ValueError):
        decode_image_bytes(b"this is not an image")
    with pytest.raises(ValueError):
        decode_image_bytes(b"")
