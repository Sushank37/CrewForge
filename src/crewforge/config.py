"""Runtime configuration, secret sanitization, and stderr-only logging for CrewForge."""

from __future__ import annotations

import json
import logging
import os
import re
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

DEFAULT_ALLOWED_CAPABILITIES: tuple[str, ...] = (
    "analysis",
    "reasoning",
    "calculation",
    "synthesis",
    "critique",
)

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{10,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[=:]\s*)([^\s,;'\"]+)", re.IGNORECASE),
)


def sanitize_error_message(message: str, max_length: int = 500) -> str:
    """Redact potential API credentials and bound raw provider error lengths."""
    cleaned = message.strip()
    for idx, pattern in enumerate(_SECRET_PATTERNS):
        if idx == 2:
            cleaned = pattern.sub(r"\1[REDACTED]", cleaned)
        else:
            cleaned = pattern.sub("[REDACTED]", cleaned)
    if len(cleaned) > max_length:
        return cleaned[: max_length - 3] + "..."
    return cleaned


class ProviderType(StrEnum):
    """Supported production model providers."""

    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class BudgetAllocationCategory(StrEnum):
    """Budget authorization categories for tracking and allocation ceilings."""

    GENERAL = "general"
    PHASE4_PILOTS = "phase4_pilots"
    PHASE5_SAME_MODEL = "phase5_same_model"
    DAVID_VS_GOLIATH = "david_vs_goliath"


class ModelPricing(BaseModel):
    """Per-1M-token USD pricing for a configured model."""

    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for the given token counts."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative.")
        return (input_tokens * self.input_per_1m + output_tokens * self.output_per_1m) / 1_000_000.0


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-6-luna": ModelPricing(input_per_1m=0.10, output_per_1m=0.50),
    "gpt-6.2-astra": ModelPricing(input_per_1m=8.00, output_per_1m=40.00),
}


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
    openai_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", min_length=1)
    ollama_num_ctx: int = Field(default=4096, ge=512, le=131072)
    ollama_think: bool | None = None

    allow_paid_runs: bool = False
    enable_http_retrieval: bool = False
    allowed_retrieval_hosts: list[str] = Field(default_factory=list)
    retrieval_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    retrieval_max_bytes: int = Field(default=262144, ge=1024, le=2_097_152)
    reference_documents: list[str] = Field(default_factory=list)
    baseline_compute_matched: bool = False
    baseline_max_calls: int | None = Field(default=None, ge=1, le=20)
    baseline_max_output_tokens: int | None = Field(default=None, ge=1, le=128000)
    baseline_max_refinement_rounds: int = Field(default=3, ge=0, le=5)
    enable_deterministic_synthesis_fast_path: bool = True

    # Evaluation-only stage switches (must remain True in production)
    enable_independent_review: bool = True
    enable_targeted_repair: bool = True
    enable_peer_messaging: bool = True

    request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    run_deadline_seconds: float = Field(default=300.0, gt=0.0)
    max_retries: int = Field(default=0, ge=0, le=10)
    max_concurrency: int = Field(default=3, ge=1, le=32)
    max_output_tokens: int = Field(default=1024, ge=1, le=128000)

    planner_max_output_tokens: int | None = Field(default=None, ge=1, le=128000)
    worker_max_output_tokens: int | None = Field(default=None, ge=1, le=128000)
    review_max_output_tokens: int | None = Field(default=None, ge=1, le=128000)
    revision_max_output_tokens: int | None = Field(default=None, ge=1, le=128000)
    synthesis_max_output_tokens: int | None = Field(default=None, ge=1, le=128000)

    max_workers: int | None = Field(default=None, ge=1, le=32)
    max_tasks: int | None = Field(default=None, ge=1, le=64)
    max_communication_rounds: int = Field(default=1, ge=0, le=3)
    max_revision_rounds: int = Field(default=1, ge=0, le=3)
    max_message_chars: int = Field(default=1000, ge=100, le=8000)
    separate_review_dimensions: bool = False
    allowed_capabilities: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_CAPABILITIES)
    )

    max_model_calls_per_run: int = Field(default=16, ge=1, le=200)
    max_cost_per_run_usd: float = Field(default=0.05, ge=0.0)
    max_cumulative_cost_usd: float = Field(default=1.00, ge=0.0)

    budget_authority_path: Path = Path(".crewforge/budget_authority.db")
    budget_allocation_category: BudgetAllocationCategory = BudgetAllocationCategory.GENERAL
    protected_reserve_usd: float = Field(default=4.00, ge=0.0)
    total_budget_ceiling_usd: float = Field(default=6.10, ge=0.0)
    phase5_same_model_budget_cap_usd: float = Field(default=0.75, ge=0.0)
    david_vs_goliath_budget_cap_usd: float = Field(default=1.10, ge=0.0)
    small_model_profile: bool = False
    cloud_reasoning_profile: bool = False

    pricing_json: dict[str, ModelPricing] = Field(default_factory=dict)

    db_path: Path = Path(".crewforge/crewforge.db")
    cloud_db_path: Path = Path(".crewforge/cloud_dev.db")
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

    @field_validator(
        "allowed_capabilities", "allowed_retrieval_hosts", "reference_documents", mode="before"
    )
    @classmethod
    def _parse_capabilities(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [part.strip() for part in stripped.split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("Field must be a list or comma-separated string.")

    @model_validator(mode="after")
    def _validate_consistency_and_secrets(self) -> CrewForgeSettings:
        if self.max_cost_per_run_usd > self.max_cumulative_cost_usd:
            raise ValueError("max_cost_per_run_usd cannot exceed max_cumulative_cost_usd.")

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

    @property
    def effective_max_concurrency(self) -> int:
        """Default local Ollama concurrency to 1 unless explicitly configured."""
        if self.small_model_profile:
            return 1
        if self.provider == ProviderType.OLLAMA and "max_concurrency" not in self.model_fields_set:
            return 1
        return self.max_concurrency

    @property
    def effective_max_workers(self) -> int:
        """Adaptive worker ceiling within configurable resource limits unless explicitly set."""
        if self.max_workers is not None:
            return self.max_workers
        available_calls = max(1, self.max_model_calls_per_run - 3)
        if self.small_model_profile:
            return max(1, min(3, available_calls))
        return min(32, max(2, available_calls))

    @property
    def effective_max_tasks(self) -> int:
        """Adaptive task ceiling within configurable resource limits unless explicitly set."""
        if self.max_tasks is not None:
            return self.max_tasks
        available_calls = max(1, self.max_model_calls_per_run - 3)
        if self.small_model_profile:
            return max(1, min(4, available_calls))
        return min(64, max(self.effective_max_workers, available_calls))

    @property
    def effective_max_revision_rounds(self) -> int:
        """Bounded revision rounds (1 for small model profile)."""
        if self.small_model_profile:
            return 1
        return self.max_revision_rounds

    def output_limit_for_operation(self, operation: str) -> int:
        """Return output-token limit for a given operation clamped to global max_output_tokens."""
        override: int | None = None
        if operation in {"planner", "plan_repair"}:
            override = self.planner_max_output_tokens
        elif operation in {"worker", "worker_followup"}:
            override = self.worker_max_output_tokens
        elif operation in {"review", "review_evidence", "review_consistency"}:
            override = self.review_max_output_tokens
        elif operation == "revision":
            override = self.revision_max_output_tokens
        elif operation == "synthesis":
            override = self.synthesis_max_output_tokens
        if override is not None:
            return min(override, self.max_output_tokens)
        return self.max_output_tokens

    @classmethod
    def for_cloud_reasoning_profile(cls, **overrides: Any) -> CrewForgeSettings:
        """Construct explicit reasoning-aware cloud profile for gpt-6-luna."""
        defaults: dict[str, Any] = {
            "provider": ProviderType.OPENAI,
            "model": "gpt-6-luna",
            "openai_reasoning_effort": "low",
            "cloud_reasoning_profile": True,
            "max_output_tokens": 8192,
            "planner_max_output_tokens": 6144,
            "worker_max_output_tokens": 8192,
            "review_max_output_tokens": 6144,
            "revision_max_output_tokens": 8192,
            "synthesis_max_output_tokens": 6144,
            "max_concurrency": 3,
            "max_retries": 0,
            "max_model_calls_per_run": 16,
            "max_cost_per_run_usd": 0.025,
            "request_timeout_seconds": 180.0,
            "run_deadline_seconds": 480.0,
        }
        defaults.update(overrides)
        return cls(**defaults)

    def get_model_pricing(self, model: str) -> ModelPricing | None:
        """Return configured pricing for a model, or default pricing, or None if unknown."""
        return self.pricing_json.get(model) or DEFAULT_MODEL_PRICING.get(model)

    @property
    def effective_db_path(self) -> Path:
        """Route paid cloud runs to cloud_db_path unless pointed to a custom DB path."""
        default_local_db = Path(".crewforge/crewforge.db")
        if self.provider in {ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE} and (
            "db_path" not in self.model_fields_set or self.db_path == default_local_db
        ):
            return self.cloud_db_path
        return self.db_path

    def to_run_config(self) -> RunConfig:
        """Create an immutable RunConfig snapshot from current settings."""
        return RunConfig(
            provider=self.provider.value,
            model=self.model,
            request_timeout_seconds=self.request_timeout_seconds,
            run_deadline_seconds=self.run_deadline_seconds,
            max_retries=self.max_retries,
            max_concurrency=self.effective_max_concurrency,
            max_output_tokens=self.max_output_tokens,
            max_model_calls_per_run=self.max_model_calls_per_run,
            max_cost_per_run_usd=self.max_cost_per_run_usd,
            max_cumulative_cost_usd=self.max_cumulative_cost_usd,
            reasoning_effort=(
                self.openai_reasoning_effort
                if self.provider in {ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE}
                else None
            ),
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
