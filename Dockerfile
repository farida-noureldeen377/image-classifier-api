# syntax=docker/dockerfile:1

# ---- Stage 1: builder ----------------------------------------------------
# Installs dependencies into an isolated virtualenv so the final image
# doesn't carry build tooling or pip caches.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create the virtualenv that we'll copy into the runtime stage.
RUN python -m venv "$VIRTUAL_ENV"

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Stage 2: runtime ----------------------------------------------------
# A lean image containing only the Python runtime, the prebuilt virtualenv,
# and the application source.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Run as a non-root user for security.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Bring over the dependencies built in the previous stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the application code.
COPY src/ ./src/

USER appuser

EXPOSE 8000

# Use the shell form so $PORT from the environment is expanded at runtime.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT}
