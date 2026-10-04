"""Backend settings. Each value can be overridden with an environment variable."""

import os

from src import config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.environ.get("DR_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", config.MODEL_FILENAME))
MODEL_INFO_PATH = os.environ.get("DR_MODEL_INFO_PATH",
                                 os.path.join(PROJECT_ROOT, "models", config.MODEL_INFO_FILENAME))

# --- Upload limits ---
MAX_FILE_SIZE_MB = float(os.environ.get("DR_MAX_FILE_SIZE_MB", "10"))   # APTOS PNGs are up to ~6 MB
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_FORMATS = {"PNG", "JPEG"}           # real format detected from the file content
MIN_IMAGE_SIDE = 64                          # smaller images cannot show retinal detail
MAX_IMAGE_PIXELS = 40_000_000                # 40 MP; largest APTOS image is ~7 MP

# --- CORS: which frontend origins may call the API ("*" = any, fine for development) ---
CORS_ORIGINS = [o.strip() for o in os.environ.get("DR_CORS_ORIGINS", "*").split(",") if o.strip()]

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend_demo")
DEMO_PAGE_PATH = os.path.join(FRONTEND_DIR, "index.html")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DISCLAIMER ="Research prototype for a college project. Not a medical diagnosis."
