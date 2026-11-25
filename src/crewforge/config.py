"""Runtime configuration and stderr-only logging setup for CrewForge."""

from __future__ import annotations

import json
import logging
import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crewforge.schemas import RunConfig

_PLACEHOLDER_KEYS = frozenset(
    {
        "",
        "your-openai-api-key-here",
        "sk-your-key-here",
        "changeme",
        "replace-me",
    }
)


class ProviderType(StrEnum):
    """Supported production model providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class ModelPricing(BaseModel):
    """Per-1M-token USD pricing for a configured model."""

    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for the given token counts."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative.")
        return (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1_000_000.0


class CrewForgeSettings(BaseSettings):
    """Application settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_prefix="CREWFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: ProviderType = ProviderType.OLLAMA
    model: str = Field(default="qwen3:4b", min_length=1)

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", min_length=1)

    request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrency: int = Field(default=3, ge=1, le=32)
    max_output_tokens: int = Field(default=1024, ge=1, le=128000)

    max_model_calls_per_run: int = Field(default=16, ge=1, le=200)
    max_cost_per_run_usd: float = Field(default=0.05, ge=0.0)
    max_cumulative_cost_usd: float = Field(default=1.00, ge=0.0)

    pricing_json: dict[str, ModelPricing] = Field(default_factory=dict)

    db_path: Path = Path(".crewforge/crewforge.db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("pricing_json", mode="before")
    @classmethod
    def _parse_pricing_json(cls, value: Any) -> dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("CREWFORGE_PRICING_JSON must decode to a JSON object.")
            return parsed
        if isinstance(value, dict):
            return value
        raise ValueError("pricing_json must be a dictionary or JSON object string.")

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _normalize_optional_url(cls, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            raise ValueError("ollama_base_url cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def _validate_consistency_and_secrets(self) -> CrewForgeSettings:
        if self.max_cost_per_run_usd > self.max_cumulative_cost_usd:
            raise ValueError(
                "max_cost_per_run_usd cannot exceed max_cumulative_cost_usd."
            )

        raw_key: str | None = None
        if self.openai_api_key is not None:
            raw_key = self.openai_api_key.get_secret_value().strip()
        elif env_key := os.environ.get("OPENAI_API_KEY"):
            raw_key = env_key.strip()

        if raw_key is not None and raw_key.lower() in _PLACEHOLDER_KEYS:
            self.openai_api_key = None
        elif raw_key:
            self.openai_api_key = SecretStr(raw_key)
        else:
            self.openai_api_key = None

        return self

    def get_model_pricing(self, model: str) -> ModelPricing | None:
        """Return configured pricing for a model, or None if unknown."""
        return self.pricing_json.get(model)

    def to_run_config(self) -> RunConfig:
        """Create an immutable RunConfig snapshot from current settings."""
        return RunConfig(
            provider=self.provider.value,
            model=self.model,
            request_timeout_seconds=self.request_timeout_seconds,
            max_retries=self.max_retries,
            max_concurrency=self.max_concurrency,
            max_output_tokens=self.max_output_tokens,
            max_model_calls_per_run=self.max_model_calls_per_run,
            max_cost_per_run_usd=self.max_cost_per_run_usd,
            max_cumulative_cost_usd=self.max_cumulative_cost_usd,
        )


def configure_logging(level: str = "INFO") -> None:
    """Configure root crewforge logger to write strictly to stderr."""
    logger = logging.getLogger("crewforge")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stderr)
