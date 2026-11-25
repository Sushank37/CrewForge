"""Ollama local provider adapter using documented HTTP /api/chat endpoints."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from crewforge.config import CrewForgeSettings
from crewforge.providers.base import (
    ModelProvider,
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    StructuredGenerationResult,
    TextGenerationResult,
    execute_with_retry,
)
from crewforge.schemas import UsageMetrics

logger = logging.getLogger("crewforge.providers.ollama")

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(ModelProvider):
    """Asynchronous adapter for local Ollama inference via /api/chat."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60.0,
        max_retries: int = 0,
        default_max_output_tokens: int = 1024,
        num_ctx: int = 4096,
        think: bool | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._default_max_output_tokens = default_max_output_tokens
        self._num_ctx = num_ctx
        self._think = think
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    @classmethod
    def from_settings(cls, settings: CrewForgeSettings) -> OllamaProvider:
        """Instantiate OllamaProvider from CrewForgeSettings."""
        return cls(
            model=settings.model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=0,
            default_max_output_tokens=settings.max_output_tokens,
            num_ctx=settings.ollama_num_ctx,
            think=settings.ollama_think,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_paid_provider(self) -> bool:
        return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _build_payload(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "num_predict": max_output_tokens or self._default_max_output_tokens,
            "num_ctx": self._num_ctx,
        }
        if temperature is not None:
            options["temperature"] = temperature

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self._think is not None:
            payload["think"] = self._think
        if json_schema is not None:
            payload["format"] = json_schema
        return payload

    async def _post_chat(self, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        try:
            response = await self._client.post(
                "/api/chat",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out: {exc}",
                provider=self.provider_name,
                model=self._model,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Failed to connect to Ollama at {self._base_url}: {exc}",
                provider=self.provider_name,
                model=self._model,
                billing_outcome="unbilled",
            ) from exc
        except httpx.HTTPError as exc:
            status_code = (
                exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            )
            retryable = status_code is None or status_code == 429 or status_code >= 500
            raise ProviderError(
                f"Ollama HTTP error (status={status_code}): {exc}",
                code="api_error",
                provider=self.provider_name,
                model=self._model,
                retryable=retryable,
                billing_outcome="unbilled",
                status_code=status_code,
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                f"Ollama returned non-JSON HTTP body: {exc}",
                code="invalid_response",
                provider=self.provider_name,
                model=self._model,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderResponseError(
                "Ollama response payload must be a JSON object.",
                code="invalid_response",
                provider=self.provider_name,
                model=self._model,
            )
        return data

    @staticmethod
    def _extract_content_and_usage(
        data: dict[str, Any],
        *,
        provider: str,
        model: str,
        retries: int,
        latency_ms: float,
    ) -> tuple[str, UsageMetrics]:
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderResponseError(
                "Ollama response missing string field 'message.content'.",
                code="invalid_response",
                provider=provider,
                model=model,
                retries_attempted=retries,
            )

        content = message["content"]
        raw_in = data.get("prompt_eval_count")
        raw_out = data.get("eval_count")
        usage_reported = raw_in is not None and raw_out is not None
        input_tokens = int(raw_in or 0)
        output_tokens = int(raw_out or 0)
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            model_calls=1,
            retries=retries,
            latency_ms=latency_ms,
            usage_reported=usage_reported,
        )
        return content, usage

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
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        data, retries, latency_ms = await execute_with_retry(
            lambda: self._post_chat(payload, timeout),
            provider=self.provider_name,
            model=self._model,
            timeout_seconds=timeout,
            max_retries=retries_cap,
        )
        content, usage = self._extract_content_and_usage(
            data,
            provider=self.provider_name,
            model=self._model,
            retries=retries,
            latency_ms=latency_ms,
        )
        if not content.strip():
            raise ProviderResponseError(
                "Ollama returned empty response content.",
                code="invalid_response",
                provider=self.provider_name,
                model=self._model,
                retries_attempted=retries,
            )

        return TextGenerationResult(
            provider=self.provider_name,
            model=self._model,
            text=content,
            usage=usage,
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
        schema = response_model.model_json_schema()
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=0.0 if temperature is None else temperature,
            json_schema=schema,
        )

        data, retries, latency_ms = await execute_with_retry(
            lambda: self._post_chat(payload, timeout),
            provider=self.provider_name,
            model=self._model,
            timeout_seconds=timeout,
            max_retries=retries_cap,
        )
        content, usage = self._extract_content_and_usage(
            data,
            provider=self.provider_name,
            model=self._model,
            retries=retries,
            latency_ms=latency_ms,
        )
        try:
            parsed = response_model.model_validate_json(content)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"Ollama structured response failed {response_model.__name__} validation: {exc}",
                code="validation_error",
                provider=self.provider_name,
                model=self._model,
                retries_attempted=retries,
            ) from exc

        logger.debug(
            "Ollama structured call succeeded (model=%s, retries=%d)",
            self._model,
            retries,
        )
        return StructuredGenerationResult[T](
            provider=self.provider_name,
            model=self._model,
            raw_text=content,
            parsed=parsed,
            usage=usage,
        )
