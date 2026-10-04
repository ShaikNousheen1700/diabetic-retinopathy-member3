"""FastAPI app: GET /health, GET /model-info, POST /predict.

Run (from the project root, venv active):
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs

Request flow for POST /predict:
    upload -> validation.validate_upload (type, size, real image) -> ModelService.predict
           -> DRPredictor (crop, resize, DenseNet121) -> JSON response
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import settings
from backend.errors import APIError, error_body
from backend.model_service import ModelService
from backend.validation import validate_upload
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend")


def create_app(service=None):
    """Build the app. Tests pass their own `service`; normally the real model is loaded."""
    service = service or ModelService(settings.MODEL_PATH, settings.MODEL_INFO_PATH)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(app):
        service.load()            # runs once when the server starts
        yield

    app = FastAPI(
        title="Diabetic Retinopathy Severity API",
        description="DenseNet121 classifier for APTOS 2019 fundus images (5 DR grades). " + settings.DISCLAIMER,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    # ---------- error handlers: every error becomes {"success": false, "error": {...}} ----------

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        log.warning("%s %s -> %d %s", request.method, request.url.path, exc.status_code, exc.code)
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content=error_body("INVALID_REQUEST", str(exc.errors())))

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(status_code=exc.status_code, content=error_body(code, str(exc.detail)))

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        log.exception("Unexpected error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=error_body("INTERNAL_ERROR", "Unexpected server error."))

    # ---------- routes ----------

    @app.get("/demo", include_in_schema=False)
    def demo_page():
        """Small upload page (frontend_demo/index.html) that calls this API — for demos and as a reference."""
        return FileResponse(settings.DEMO_PAGE_PATH, media_type="text/html")

    @app.get("/health")
    def health():
        """Is the server up, and is the model ready?"""
        return {
            "status": "ok" if service.loaded else "degraded",
            "model_loaded": service.loaded,
            "uptime_seconds": round(time.time() - started_at, 1),
        }

    @app.get("/model-info")
    def model_info():
        """Which model is served and how inputs are handled."""
        return {
            "model_loaded": service.loaded,
            "load_error": service.load_error,
            "model": service.info.get("model_name", "DenseNet121"),
            "framework": service.info.get("framework"),
            "input_size": service.info.get("input_size", [config.IMG_SIZE, config.IMG_SIZE, 3]),
            "classes": {i: name for i, name in enumerate(config.CLASS_NAMES)},
            "preprocessing": service.info.get("preprocessing"),
            "training": {k: service.info.get(k) for k in ("weights", "split_sizes", "epochs_trained", "trained_at")},
            "upload_limits": {
                "allowed_types": sorted(settings.ALLOWED_EXTENSIONS),
                "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
                "min_image_side": settings.MIN_IMAGE_SIDE,
                "max_image_megapixels": settings.MAX_IMAGE_PIXELS / 1e6,
            },
            "disclaimer": settings.DISCLAIMER,
        }

    # `def` (not `async def`): FastAPI runs it in a worker thread, so slow model inference
    # does not block the server from answering other requests such as /health.
    @app.post("/predict")
    def predict(file: UploadFile | None = File(None, description="Fundus image (PNG or JPEG)")):
        """Upload one fundus image; returns the predicted DR grade and class probabilities."""
        t0 = time.time()
        if file is None:
            raise APIError(400, "MISSING_FILE", "No file uploaded. Send the image in a form field named 'file'.")

        # read at most limit+1 bytes: enough to know the file is too large without reading all of it
        data = file.file.read(settings.MAX_FILE_SIZE_BYTES + 1)
        img = validate_upload(file.filename, data)
        prediction = service.predict(img)

        elapsed_ms = round(1000 * (time.time() - t0), 1)
        log.info("predict %s (%dx%d) -> %s %.3f in %.0f ms", file.filename, img.shape[1], img.shape[0],
                 prediction["class_name"], prediction["confidence"], elapsed_ms)
        return {
            "success": True,
            "filename": file.filename,
            "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
            "prediction": prediction,
            "model": "DenseNet121",
            "processing_time_ms": elapsed_ms,
            "disclaimer": settings.DISCLAIMER,
        }

    return app


app = create_app()
