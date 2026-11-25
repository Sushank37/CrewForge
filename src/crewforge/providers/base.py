"""Asynchronous model provider interface, structured errors, and bounded retry helpers."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from crewforge.config import sanitize_error_message
from crewforge.schemas import UsageMetrics

T = TypeVar("T", bound=BaseModel)

BillingOutcome = Literal["unbilled", "unknown", "billed"]


class ProviderError(Exception):
    """Structured error raised by model providers."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: str,
        model: str,
        retryable: bool = False,
        billing_outcome: BillingOutcome = "unknown",
        status_code: int | None = None,
        retries_attempted: int = 0,
        usage: UsageMetrics | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        safe_message = sanitize_error_message(message)
        super().__init__(safe_message)
        self.message = safe_message
        self.code = code
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.billing_outcome = billing_outcome
        self.status_code = status_code
        self.retries_attempted = retries_attempted
        self.usage = usage
        self.response_metadata = dict(response_metadata) if response_metadata else {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize sanitized error details for logging or event payloads."""
        data: dict[str, Any] = {
            "code": self.code,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
            "retryable": self.retryable,
            "billing_outcome": self.billing_outcome,
            "status_code": self.status_code,
            "retries_attempted": self.retries_attempted,
        }
        if self.usage is not None:
            data["usage"] = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "reasoning_tokens": self.usage.reasoning_tokens,
                "cached_input_tokens": self.usage.cached_input_tokens,
                "total_tokens": self.usage.total_tokens,
                "usage_reported": self.usage.usage_reported,
                "returned_model": self.usage.returned_model,
                "provider_response_id": self.usage.provider_response_id,
                "response_status": self.usage.response_status,
                "incomplete_reason": self.usage.incomplete_reason,
            }
        if self.response_metadata:
            data["response_metadata"] = self.response_metadata
        return data


class ProviderConfigurationError(ProviderError):
    """Raised when provider credentials or configuration are invalid."""

    def __init__(self, message: str, *, provider: str, model: str) -> None:
        super().__init__(
            message,
            code="configuration_error",
            provider=provider,
            model=model,
            retryable=False,
            billing_outcome="unbilled",
        )


class ProviderTimeoutError(ProviderError):
    """Raised when a model request exceeds its timeout."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        retries_attempted: int = 0,
    ) -> None:
        super().__init__(
            message,
            code="timeout",
            provider=provider,
            model=model,
            retryable=True,
            billing_outcome="unknown",
            retries_attempted=retries_attempted,
        )


class ProviderRateLimitError(ProviderError):
    """Raised when an upstream provider rejects a call due to rate limiting."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        status_code: int | None = 429,
        retries_attempted: int = 0,
    ) -> None:
        super().__init__(
            message,
            code="rate_limit",
            provider=provider,
            model=model,
            retryable=True,
            billing_outcome="unbilled",
            status_code=status_code,
            retries_attempted=retries_attempted,
        )


class ProviderConnectionError(ProviderError):
    """Raised on network connection failures to the provider endpoint."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        billing_outcome: BillingOutcome = "unknown",
        retries_attempted: int = 0,
    ) -> None:
        super().__init__(
            message,
            code="connection_error",
            provider=provider,
            model=model,
            retryable=True,
            billing_outcome=billing_outcome,
            retries_attempted=retries_attempted,
        )


class ProviderResponseError(ProviderError):
    """Raised when a model response is malformed or fails Pydantic validation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_response",
        provider: str,
        model: str,
        billing_outcome: BillingOutcome = "billed",
        retries_attempted: int = 0,
        usage: UsageMetrics | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            provider=provider,
            model=model,
            retryable=False,
            billing_outcome=billing_outcome,
            retries_attempted=retries_attempted,
            usage=usage,
            response_metadata=response_metadata,
        )


class TextGenerationResult(BaseModel):
    """Normalized result from an asynchronous text generation request."""

    provider: str
    model: str
    returned_model: str | None = None
    text: str
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    raw_response_id: str | None = None


class StructuredGenerationResult[T: BaseModel](BaseModel):
    """Normalized result from an asynchronous structured JSON request."""

    provider: str
    model: str
    returned_model: str | None = None
    raw_text: str
    parsed: T
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    raw_response_id: str | None = None


class ModelProvider(ABC):
    """Common asynchronous interface for all LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical provider identifier (e.g., 'openai', 'ollama')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""

    @property
    @abstractmethod
    def is_paid_provider(self) -> bool:
        """Whether calls incur external metered monetary costs."""

    @property
    def supports_temperature(self) -> bool:
        """Whether `temperature` parameter is supported by this provider/model configuration."""
        return True

    @property
    def supports_reasoning_effort(self) -> bool:
        """Whether `reasoning.effort` parameter is supported."""
        return False

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> TextGenerationResult:
        """Generate an unstructured text completion."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> StructuredGenerationResult[T]:
        """Generate and validate a Pydantic-structured JSON response."""

    async def aclose(self) -> None:
        """Release underlying HTTP/SDK client resources."""
        return None


async def execute_with_retry[R](
    operation: Callable[[], Awaitable[R]],
    *,
    provider: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
    base_backoff_seconds: float = 0.25,
) -> tuple[R, int, float]:
    """Execute an async provider call with timeout, cancellation, and bounded retries.

    Returns (result, retries_attempted, total_latency_ms).
    """
    start = time.perf_counter()
    attempt = 0

    while True:
        try:
            result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return result, attempt, elapsed_ms
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            err = ProviderTimeoutError(
                f"Request to {provider}/{model} timed out after {timeout_seconds:.2f}s.",
                provider=provider,
                model=model,
                retries_attempted=attempt,
            )
            if attempt >= max_retries:
                raise err from exc
        except ProviderError as err:
            err.retries_attempted = attempt
            if not err.retryable or attempt >= max_retries:
                raise

        attempt += 1
        await asyncio.sleep(base_backoff_seconds * (2 ** (attempt - 1)))
