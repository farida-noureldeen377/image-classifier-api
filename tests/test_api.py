"""API tests.

These tests run without downloading ResNet50: the lifespan handler that
loads the real model is replaced with a lightweight fake classifier. Test
images are generated in-memory with PIL so nothing is read from disk.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import pytest

# Config validates the environment at import time, so make sure the required
# variables exist before `src` is imported.
os.environ.setdefault("PORT", "8000")
os.environ.setdefault("MODEL_CONFIDENCE_THRESHOLD", "0.5")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from src import api  # noqa: E402


class FakeClassifier:
    """Stand-in for the real Classifier — no model download, deterministic."""

    def predict(self, image_bytes: bytes) -> list[dict]:
        return [{"label": "test_object", "confidence": 0.99}]


@pytest.fixture
def client():
    """TestClient whose lifespan installs the FakeClassifier."""

    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.classifier = FakeClassifier()
        app.state.model_ready = True
        yield
        app.state.classifier = None
        app.state.model_ready = False

    # Swap the lifespan, then restore it afterwards so tests stay isolated.
    original = api.app.router.lifespan_context
    api.app.router.lifespan_context = fake_lifespan
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        api.app.router.lifespan_context = original


def _make_image_bytes(fmt: str = "JPEG") -> bytes:
    """Create a small solid-color image in-memory and return its bytes."""
    image = Image.new("RGB", (32, 32), color=(120, 80, 200))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert "message" in body


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_predict_with_valid_image(client):
    image_bytes = _make_image_bytes("JPEG")
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "test.jpg"
    assert body["predictions"] == [{"label": "test_object", "confidence": 0.99}]


def test_predict_rejects_non_image_file(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_rejects_empty_file(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
