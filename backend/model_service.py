"""Holds the DenseNet121 model for the lifetime of the server.

The model is loaded ONCE at startup (≈10 s) — not per request — so each prediction only
costs the inference time (≈0.3 s on CPU).

If loading fails (e.g. the .keras file is missing), the server still starts:
/health reports model_loaded=false and /predict answers 503, instead of the whole server crashing.
"""

import json
import logging
import os
import threading
import time

import numpy as np

from backend.errors import APIError

log = logging.getLogger("backend")


class ModelService:
    def __init__(self, model_path, info_path=None, predictor=None):
        """predictor: optional ready-made object with predict_rgb() — used by tests."""
        self.model_path = model_path
        self.info_path = info_path
        self.predictor = predictor
        self.load_error = None
        self.info = {}
        self.loaded_at = None
        # FastAPI runs requests in parallel threads; the lock lets only one use the model at a time.
        self._lock = threading.Lock()

    @property
    def loaded(self):
        return self.predictor is not None

    def load(self):
        """Load model + info file. Errors are stored, not raised."""
        if self.info_path and os.path.exists(self.info_path):
            with open(self.info_path) as f:
                self.info = json.load(f)
        if self.predictor is None:
            try:
                from src.predict import DRPredictor      # imported here: TensorFlow is slow to import
                t0 = time.time()
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                self.predictor = DRPredictor(self.model_path)
                # Warm-up: TensorFlow builds its computation graph on the first call (~3 s).
                # Doing it here keeps the first real user request fast (~0.3 s).
                self.predictor.predict_rgb(np.zeros((256, 256, 3), dtype=np.uint8))
                log.info("Model loaded and warmed up from %s in %.1fs", self.model_path, time.time() - t0)
            except Exception as err:
                self.load_error = f"{type(err).__name__}: {err}"
                log.error("Model could not be loaded: %s", self.load_error)
                return
        self.loaded_at = time.time()

    def predict(self, img):
        """img: RGB numpy array. Returns the DRPredictor result dict."""
        if not self.loaded:
            raise APIError(503, "MODEL_NOT_LOADED", "The model is not loaded. Check the server logs.")
        try:
            with self._lock:
                return self.predictor.predict_rgb(img)
        except Exception:
            log.exception("Prediction failed")
            raise APIError(500, "PREDICTION_FAILED", "The model failed to process this image.")
