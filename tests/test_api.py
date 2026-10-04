"""Backend API tests.

Part 1 uses a FAKE predictor (fast, and lets us simulate model failures).
Part 2 uses the REAL trained DenseNet121 (skipped if models/densenet121_dr.keras is missing).
"""

import io
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import settings
from backend.main import create_app
from backend.model_service import ModelService
from src import config
from tests.conftest import make_fundus

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_MODEL = os.path.join(PROJECT_ROOT, "models", config.MODEL_FILENAME)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")


# ---------------------------------------------------------------- helpers

def encode(img_rgb, ext=".png"):
    ok, buf = cv2.imencode(ext, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def png_header_only(width, height):
    """A tiny PNG whose header claims a huge size (no pixel data) — tests the pixel limit cheaply."""
    def chunk(kind, payload=b""):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)   # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT") + chunk(b"IEND")


def upload(client, data, filename="eye.png", field="file", content_type="image/png"):
    return client.post("/predict", files={field: (filename, data, content_type)})


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


FAKE_RESULT = {"class_id": 2, "class_name": "Moderate", "confidence": 0.81,
               "probabilities": {"No DR": 0.02, "Mild": 0.05, "Moderate": 0.81, "Severe": 0.08,
                                 "Proliferative DR": 0.04}}


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict_rgb(self, img):
        self.calls += 1
        return dict(FAKE_RESULT)


class BrokenPredictor:
    def predict_rgb(self, img):
        raise RuntimeError("simulated model crash")


@pytest.fixture
def client():
    service = ModelService("unused.keras", predictor=FakePredictor())
    with TestClient(create_app(service)) as c:      # `with` runs startup (model load)
        yield c


# ---------------------------------------------------------------- health / model-info

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "DenseNet121"
    assert body["classes"] == {str(i): n for i, n in enumerate(config.CLASS_NAMES)}
    assert body["upload_limits"]["allowed_types"] == [".jpeg", ".jpg", ".png"]


# ---------------------------------------------------------------- valid uploads + response structure

def test_predict_valid_png_response_structure(client):
    r = upload(client, encode(make_fundus()))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"success", "filename", "image_size", "prediction", "model",
                         "processing_time_ms", "disclaimer"}
    assert body["success"] is True
    assert body["image_size"] == {"width": 400, "height": 300}
    assert set(body["prediction"]) == {"class_id", "class_name", "confidence", "probabilities"}
    assert body["prediction"]["class_name"] == "Moderate"


def test_predict_valid_jpeg(client):
    r = upload(client, encode(make_fundus(), ".jpg"), filename="eye.JPG", content_type="image/jpeg")
    assert r.status_code == 200 and r.json()["success"] is True


# ---------------------------------------------------------------- missing file

def test_missing_file_no_body(client):
    assert_error(client.post("/predict"), 400, "MISSING_FILE")


def test_missing_file_wrong_field_name(client):
    assert_error(upload(client, encode(make_fundus()), field="image"), 400, "MISSING_FILE")


def test_empty_file(client):
    assert_error(upload(client, b""), 400, "EMPTY_FILE")


# ---------------------------------------------------------------- unsupported file type

def test_unsupported_extension(client):
    assert_error(upload(client, b"hello", filename="notes.txt", content_type="text/plain"),
                 415, "UNSUPPORTED_FILE_TYPE")


def test_gif_renamed_to_png(client):
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(buf, format="GIF")
    assert_error(upload(client, buf.getvalue(), filename="eye.png"), 415, "UNSUPPORTED_FILE_TYPE")


# ---------------------------------------------------------------- corrupted image

def test_text_file_renamed_to_png(client):
    assert_error(upload(client, b"this is not an image at all" * 10), 422, "CORRUPTED_IMAGE")


def test_truncated_png(client):
    data = encode(make_fundus())
    assert_error(upload(client, data[: len(data) // 2]), 422, "CORRUPTED_IMAGE")


def test_truncated_jpeg(client):
    data = encode(make_fundus(), ".jpg")
    assert_error(upload(client, data[:200], filename="eye.jpg", content_type="image/jpeg"),
                 422, "CORRUPTED_IMAGE")


# ---------------------------------------------------------------- oversized / undersized

def test_file_too_large(client):
    data = b"\x89PNG\r\n\x1a\n" + b"\0" * settings.MAX_FILE_SIZE_BYTES
    assert_error(upload(client, data), 413, "FILE_TOO_LARGE")


def test_image_resolution_too_large(client):
    assert_error(upload(client, png_header_only(10_000, 10_000)), 413, "IMAGE_TOO_LARGE")


def test_image_too_small(client):
    assert_error(upload(client, encode(make_fundus(40, 40))), 422, "IMAGE_TOO_SMALL")


# ---------------------------------------------------------------- model / server errors

def test_model_crash_returns_500():
    service = ModelService("unused.keras", predictor=BrokenPredictor())
    with TestClient(create_app(service)) as c:
        assert_error(upload(c, encode(make_fundus())), 500, "PREDICTION_FAILED")
        assert c.get("/health").status_code == 200          # server keeps running


def test_model_file_missing_returns_503():
    service = ModelService("/nonexistent/model.keras")
    with TestClient(create_app(service)) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        assert "FileNotFoundError" in c.get("/model-info").json()["load_error"]
        assert_error(upload(c, encode(make_fundus())), 503, "MODEL_NOT_LOADED")


def test_wrong_method_and_unknown_route(client):
    assert_error(client.get("/predict"), 405, "METHOD_NOT_ALLOWED")
    assert_error(client.get("/nope"), 404, "NOT_FOUND")


# ---------------------------------------------------------------- repeated requests

def test_repeated_requests(client):
    data = encode(make_fundus())
    results = [upload(client, data).json()["prediction"] for _ in range(20)]
    assert all(r == results[0] for r in results)
    assert client.app.state.service.predictor.calls == 20


def test_concurrent_requests(client):
    data = encode(make_fundus())
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: upload(client, data), range(16)))
    assert all(r.status_code == 200 for r in responses)


def test_error_then_success(client):
    """A bad request must not break the next good one."""
    assert upload(client, b"garbage").status_code == 422
    assert upload(client, encode(make_fundus())).status_code == 200


# ---------------------------------------------------------------- real DenseNet121

real = pytest.mark.skipif(not os.path.exists(REAL_MODEL), reason="trained model not in models/")


@pytest.fixture(scope="module")
def real_client():
    with TestClient(create_app(ModelService(REAL_MODEL, settings.MODEL_INFO_PATH))) as c:
        yield c


@real
def test_real_model_info(real_client):
    body = real_client.get("/model-info").json()
    assert body["model_loaded"] is True
    assert body["training"]["split_sizes"] == {"train": 2564, "val": 549, "test": 549}


@real
def test_real_model_prediction_matches_direct_predictor(real_client):
    """API result must equal calling DRPredictor directly — proves identical preprocessing."""
    from src.predict import DRPredictor
    samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(".png")) if os.path.isdir(SAMPLES_DIR) else []
    if not samples:
        pytest.skip("no sample fundus images in samples/")
    direct = DRPredictor(REAL_MODEL)
    for name in samples:
        path = os.path.join(SAMPLES_DIR, name)
        with open(path, "rb") as f:
            api = upload(real_client, f.read(), filename=name).json()
        assert api["success"] is True
        assert api["prediction"] == direct.predict_path(path)
        assert abs(sum(api["prediction"]["probabilities"].values()) - 1) < 1e-3


@real
def test_real_model_repeated_requests_consistent(real_client):
    data = encode(make_fundus())
    first = upload(real_client, data).json()["prediction"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        others = list(pool.map(lambda _: upload(real_client, data).json()["prediction"], range(8)))
    assert all(o == first for o in others)
