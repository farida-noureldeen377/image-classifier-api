"""FastAPI application for the image classifier service.

Exposes three endpoints:

* ``GET  /``        – welcome message and link to the docs.
* ``GET  /health``  – liveness/readiness probe.
* ``POST /predict`` – accepts an uploaded image, returns predictions.

The model is loaded once on startup via the lifespan handler and stored on
``app.state`` so requests share a single instance.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .config import config
from .model import Classifier, ImageDecodeError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Content types we accept. Anything else is rejected with a 400.
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, drop the reference on shutdown."""
    logger.info("Starting up: loading classifier...")
    try:
        app.state.classifier = Classifier(
            confidence_threshold=config.model_confidence_threshold
        )
        app.state.model_ready = True
    except Exception:  # noqa: BLE001 - we want startup failures logged loudly
        # Keep the app up so /health can report the failure, but mark the
        # model as unavailable. /predict will then return 503.
        app.state.classifier = None
        app.state.model_ready = False
        logger.exception("Failed to load the classifier at startup.")
    yield
    app.state.classifier = None
    app.state.model_ready = False


app = FastAPI(
    title="Image Classifier API",
    description="Upload an image and get back the objects it likely contains.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict:
    """Welcome message with a pointer to the interactive docs."""
    return {
        "message": "Image Classifier API. POST an image to /predict.",
        "docs": "/docs",
    }


@app.get("/health")
async def health(request: Request) -> dict:
    """Report service status and whether the model loaded successfully."""
    return {
        "status": "ok",
        "model_ready": bool(getattr(request.app.state, "model_ready", False)),
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict:
    """Classify an uploaded image and return the predicted objects."""
    classifier: Classifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Check service logs.",
        )

    # Reject anything that isn't a supported image type up front.
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}."
            ),
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        predictions = classifier.predict(image_bytes)
    except ImageDecodeError:
        # Content type claimed image but bytes weren't a valid/decodable image.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be read as a valid image.",
        )
    except Exception:  # noqa: BLE001 - surface inference failures as 500
        logger.exception("Inference failed for uploaded file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run inference on the image.",
        )

    return {"filename": file.filename, "predictions": predictions}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort handler so unexpected errors return clean JSON."""
    logger.exception("Unhandled error processing %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
