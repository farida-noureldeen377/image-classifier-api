"""Application configuration.

Loads settings from the environment (and a local ``.env`` file), validates
them, and exposes a single immutable ``config`` object for the rest of the
app to import. If anything is missing or invalid, importing this module
fails loudly with an actionable message rather than letting a bad value
surface deep inside request handling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file if present. Real environment
# variables already set take precedence (override=False), which is what we
# want in Docker/production where config comes from the environment.
load_dotenv(override=False)


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Validated, read-only application settings."""

    port: int
    model_confidence_threshold: float


def _require(name: str) -> str:
    """Return the env var ``name`` or raise a clear ConfigError if unset/empty."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and set it."
        )
    return value.strip()


def _parse_port(raw: str) -> int:
    try:
        port = int(raw)
    except ValueError:
        raise ConfigError(
            f"PORT must be an integer, got '{raw}'."
        ) from None
    if not (1 <= port <= 65535):
        raise ConfigError(f"PORT must be between 1 and 65535, got {port}.")
    return port


def _parse_threshold(raw: str) -> float:
    try:
        threshold = float(raw)
    except ValueError:
        raise ConfigError(
            f"MODEL_CONFIDENCE_THRESHOLD must be a number, got '{raw}'."
        ) from None
    if not (0.0 <= threshold <= 1.0):
        raise ConfigError(
            f"MODEL_CONFIDENCE_THRESHOLD must be a float between 0 and 1, "
            f"got {threshold}."
        )
    return threshold


def load_config() -> Config:
    """Build and validate a Config from the environment."""
    return Config(
        port=_parse_port(_require("PORT")),
        model_confidence_threshold=_parse_threshold(
            _require("MODEL_CONFIDENCE_THRESHOLD")
        ),
    )


try:
    config = load_config()
except ConfigError as exc:
    # Fail fast at import time with a clear, single-line message and a
    # non-zero exit code so misconfiguration is obvious on startup.
    print(f"[config] Configuration error: {exc}", file=sys.stderr)
    raise SystemExit(1)
