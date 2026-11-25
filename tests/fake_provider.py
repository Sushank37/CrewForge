"""Deterministic fake ModelProvider strictly for offline automated testing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from crewforge.providers.base import (
    ModelProvider,
    ProviderResponseError,
    StructuredGenerationResult,
    TextGenerationResult,
    execute_with_retry,
)
from crewforge.schemas import UsageMetrics

T = TypeVar("T", bound=BaseModel)


class FakeProvider(ModelProvider):
    """Deterministic, zero-cost test provider with scriptable outcomes."""

    def __init__(
        self,
        *,
        provider_name: str = "fake",
        model_name: str = "fake-model-v1",
        is_paid_provider: bool = False,
        timeout_seconds: float = 5.0,
        max_retries: int = 1,
        handler: Callable[[str, type[BaseModel] | None], Any] | None = None,
        delay_seconds: float = 0.0,
        input_tokens: int = 42,
        output_tokens: int = 24,
        usage_reported: bool = True,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self._is_paid = is_paid_provider
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._handler = handler
        self._delay_seconds = delay_seconds
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._usage_reported = usage_reported
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_paid_provider(self) -> bool:
        return self._is_paid

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
        timeout = timeout_seconds or self._timeout_seconds
        retries_cap = self._max_retries if max_retries is None else max_retries
        self.calls.append({"mode": "text", "prompt": prompt, "system_prompt": system_prompt})

        async def _call() -> str:
            if self._delay_seconds > 0:
                await asyncio.sleep(self._delay_seconds)
            if self._handler is not None:
                res = self._handler(prompt, None)
                if isinstance(res, Exception):
                    raise res
                return str(res)
            return f"Deterministic response to: {prompt}"

        text, retries, latency_ms = await execute_with_retry(
            _call,
            provider=self.provider_name,
            model=self.model_name,
            timeout_seconds=timeout,
            max_retries=retries_cap,
            base_backoff_seconds=0.01,
        )
        return TextGenerationResult(
            provider=self.provider_name,
            model=self.model_name,
            text=text,
            usage=UsageMetrics(
                input_tokens=self._input_tokens if self._usage_reported else 0,
                output_tokens=self._output_tokens if self._usage_reported else 0,
                total_tokens=(
                    (self._input_tokens + self._output_tokens) if self._usage_reported else 0
                ),
                model_calls=1,
                retries=retries,
                latency_ms=latency_ms,
                usage_reported=self._usage_reported,
            ),
        )

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
        timeout = timeout_seconds or self._timeout_seconds
        retries_cap = self._max_retries if max_retries is None else max_retries
        self.calls.append(
            {
                "mode": "structured",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_model": response_model.__name__,
            }
        )

        async def _call() -> Any:
            if self._delay_seconds > 0:
                await asyncio.sleep(self._delay_seconds)
            if self._handler is not None:
                res = self._handler(prompt, response_model)
                if isinstance(res, Exception):
                    raise res
                return res
            return {
                "answer": f"Deterministic baseline answer for: {prompt}",
                "missing_information": ["No live runtime environment metrics were provided."],
                "assumptions": ["Evaluated offline using deterministic test provider."],
                "confidence_notes": "Deterministic test output.",
            }

        raw_or_obj, retries, latency_ms = await execute_with_retry(
            _call,
            provider=self.provider_name,
            model=self.model_name,
            timeout_seconds=timeout,
            max_retries=retries_cap,
            base_backoff_seconds=0.01,
        )

        if isinstance(raw_or_obj, response_model):
            parsed = raw_or_obj
            raw_text = parsed.model_dump_json()
        elif isinstance(raw_or_obj, str):
            raw_text = raw_or_obj
            try:
                parsed = response_model.model_validate_json(raw_text)
            except ValidationError as exc:
                raise ProviderResponseError(
                    f"FakeProvider received invalid structured JSON: {exc}",
                    code="validation_error",
                    provider=self.provider_name,
                    model=self.model_name,
                    retries_attempted=retries,
                ) from exc
        else:
            try:
                parsed = response_model.model_validate(raw_or_obj)
                raw_text = parsed.model_dump_json()
            except ValidationError as exc:
                raise ProviderResponseError(
                    f"FakeProvider received invalid structured payload: {exc}",
                    code="validation_error",
                    provider=self.provider_name,
                    model=self.model_name,
                    retries_attempted=retries,
                ) from exc

        return StructuredGenerationResult[T](
            provider=self.provider_name,
            model=self.model_name,
            raw_text=raw_text,
            parsed=parsed,
            usage=UsageMetrics(
                input_tokens=self._input_tokens if self._usage_reported else 0,
                output_tokens=self._output_tokens if self._usage_reported else 0,
                total_tokens=(
                    (self._input_tokens + self._output_tokens) if self._usage_reported else 0
                ),
                model_calls=1,
                retries=retries,
                latency_ms=latency_ms,
                usage_reported=self._usage_reported,
            ),
        )
