"""OpenAI provider adapter using the official AsyncOpenAI Responses API."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.providers.base import (
    ModelProvider,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    StructuredGenerationResult,
    TextGenerationResult,
    execute_with_retry,
)
from crewforge.schemas import UsageMetrics

logger = logging.getLogger("crewforge.providers.openai")

T = TypeVar("T", bound=BaseModel)


_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    norm = model.strip().lower()
    return norm.startswith(_REASONING_MODEL_PREFIXES)


class OpenAIProvider(ModelProvider):
    """Asynchronous OpenAI adapter backed by Responses API (or Chat Completions for compat)."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        use_chat_completions: bool = False,
        provider_label: str = "openai",
        timeout_seconds: float = 60.0,
        max_retries: int = 0,
        default_max_output_tokens: int = 1024,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if client is None and (api_key is None or not api_key.strip()):
            raise ProviderConfigurationError(
                "OpenAI provider requires a non-empty API key (CREWFORGE_OPENAI_API_KEY).",
                provider=provider_label,
                model=model,
            )
        if use_chat_completions and client is None and (base_url is None or not base_url.strip()):
            raise ProviderConfigurationError(
                "OpenAI-compatible provider requires CREWFORGE_OPENAI_BASE_URL.",
                provider=provider_label,
                model=model,
            )
        self._model = model
        self._provider_label = provider_label
        self._reasoning_effort = reasoning_effort
        self._use_chat_completions = use_chat_completions
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._default_max_output_tokens = default_max_output_tokens
        self._owns_client = client is None
        # Disable SDK-level hidden retries so BudgetManager/ModelExecutor owns all attempts.
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    @classmethod
    def from_settings(cls, settings: CrewForgeSettings) -> OpenAIProvider:
        """Instantiate OpenAIProvider from CrewForgeSettings."""
        secret = (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key is not None
            else None
        )
        is_compat = settings.provider == ProviderType.OPENAI_COMPATIBLE
        return cls(
            model=settings.model,
            api_key=secret,
            base_url=settings.openai_base_url,
            reasoning_effort=None if is_compat else settings.openai_reasoning_effort,
            use_chat_completions=is_compat,
            provider_label=settings.provider.value,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=0,
            default_max_output_tokens=settings.max_output_tokens,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_label

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_paid_provider(self) -> bool:
        return True

    @property
    def supports_reasoning_effort(self) -> bool:
        return (not self._use_chat_completions) and _is_reasoning_model(self._model)

    @property
    def supports_temperature(self) -> bool:
        if _is_reasoning_model(self._model):
            return self._reasoning_effort == "none"
        return True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    def _translate_sdk_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, ValidationError):
            return ProviderResponseError(
                f"Structured output validation failed: {exc}",
                code="validation_error",
                provider=self.provider_name,
                model=self._model,
            )
        length_err = getattr(openai, "LengthFinishReasonError", None)
        if length_err is not None and isinstance(exc, length_err):
            return ProviderResponseError(
                "OpenAI structured response truncated at max_output_tokens limit.",
                code="truncated_response",
                provider=self.provider_name,
                model=self._model,
            )
        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(
                f"OpenAI request timed out: {exc}",
                provider=self.provider_name,
                model=self._model,
            )
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimitError(
                f"OpenAI rate limit error: {exc}",
                provider=self.provider_name,
                model=self._model,
                status_code=getattr(exc, "status_code", 429),
            )
        if isinstance(exc, openai.AuthenticationError):
            return ProviderConfigurationError(
                "OpenAI authentication failed (check API key).",
                provider=self.provider_name,
                model=self._model,
            )
        if isinstance(exc, openai.APIConnectionError):
            return ProviderConnectionError(
                f"OpenAI connection error: {exc}",
                provider=self.provider_name,
                model=self._model,
                billing_outcome="unknown",
            )
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            retryable = status is not None and (status == 429 or status >= 500)
            billing = "unbilled" if status is not None and 400 <= status < 500 else "unknown"
            return ProviderError(
                f"OpenAI API error (status={status}): {exc}",
                code="api_error",
                provider=self.provider_name,
                model=self._model,
                retryable=retryable,
                billing_outcome=billing,
                status_code=status,
            )
        return ProviderError(
            f"Unexpected OpenAI error: {exc}",
            code="api_error",
            provider=self.provider_name,
            model=self._model,
            retryable=False,
            billing_outcome="unknown",
        )

    @staticmethod
    def _extract_usage(
        response: Any,
        retries: int,
        latency_ms: float,
        *,
        fallback_model: str | None = None,
    ) -> UsageMetrics:
        returned_model = getattr(response, "model", None) or fallback_model
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return UsageMetrics(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model_calls=1,
                retries=retries,
                latency_ms=latency_ms,
                usage_reported=False,
                returned_model=returned_model,
            )

        raw_in = getattr(usage_obj, "input_tokens", None)
        if raw_in is None:
            raw_in = getattr(usage_obj, "prompt_tokens", None)
        raw_out = getattr(usage_obj, "output_tokens", None)
        if raw_out is None:
            raw_out = getattr(usage_obj, "completion_tokens", None)
        if raw_in is None or raw_out is None:
            return UsageMetrics(
                input_tokens=int(raw_in or 0),
                output_tokens=int(raw_out or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
                model_calls=1,
                retries=retries,
                latency_ms=latency_ms,
                usage_reported=False,
                returned_model=returned_model,
            )

        input_tokens = int(raw_in)
        output_tokens = int(raw_out)
        raw_total = getattr(usage_obj, "total_tokens", None)
        total_tokens = int(raw_total) if raw_total is not None else (input_tokens + output_tokens)
        return UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=max(total_tokens, input_tokens + output_tokens),
            model_calls=1,
            retries=retries,
            latency_ms=latency_ms,
            usage_reported=True,
            returned_model=returned_model,
        )

    def _build_common_kwargs(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": prompt,
            "max_output_tokens": max_output_tokens or self._default_max_output_tokens,
        }
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if self._reasoning_effort is not None and self.supports_reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        if temperature is not None and self.supports_temperature:
            kwargs["temperature"] = temperature
        return kwargs

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

        if self._use_chat_completions:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            chat_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_output_tokens or self._default_max_output_tokens,
            }
            if temperature is not None and self.supports_temperature:
                chat_kwargs["temperature"] = temperature

            async def _chat_call() -> Any:
                try:
                    return await self._client.chat.completions.create(**chat_kwargs)
                except Exception as exc:
                    raise self._translate_sdk_error(exc) from exc

            response, retries, latency_ms = await execute_with_retry(
                _chat_call,
                provider=self.provider_name,
                model=self._model,
                timeout_seconds=timeout,
                max_retries=retries_cap,
            )
            choices = getattr(response, "choices", None) or []
            output_text = getattr(choices[0].message, "content", None) if choices else None
            if not isinstance(output_text, str) or not output_text.strip():
                raise ProviderResponseError(
                    "OpenAI-compatible endpoint returned empty chat content.",
                    provider=self.provider_name,
                    model=self._model,
                    retries_attempted=retries,
                )
            usage = self._extract_usage(response, retries, latency_ms, fallback_model=self._model)
            return TextGenerationResult(
                provider=self.provider_name,
                model=self._model,
                returned_model=usage.returned_model,
                text=output_text,
                usage=usage,
                raw_response_id=getattr(response, "id", None),
            )

        kwargs = self._build_common_kwargs(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        async def _call() -> Any:
            try:
                return await self._client.responses.create(**kwargs)
            except Exception as exc:
                raise self._translate_sdk_error(exc) from exc

        response, retries, latency_ms = await execute_with_retry(
            _call,
            provider=self.provider_name,
            model=self._model,
            timeout_seconds=timeout,
            max_retries=retries_cap,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ProviderResponseError(
                "OpenAI Responses API returned empty output_text.",
                provider=self.provider_name,
                model=self._model,
                retries_attempted=retries,
            )

        usage = self._extract_usage(response, retries, latency_ms, fallback_model=self._model)
        return TextGenerationResult(
            provider=self.provider_name,
            model=self._model,
            returned_model=usage.returned_model,
            text=output_text,
            usage=usage,
            raw_response_id=getattr(response, "id", None),
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

        if self._use_chat_completions:
            import json as _json

            schema_hint = _json.dumps(response_model.model_json_schema(), separators=(",", ":"))
            sys_msg = (
                system_prompt or ""
            ) + f"\nReturn strictly valid JSON matching schema: {schema_hint}"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": sys_msg.strip()},
                {"role": "user", "content": prompt},
            ]
            chat_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_output_tokens or self._default_max_output_tokens,
                "response_format": {"type": "json_object"},
            }
            if temperature is not None and self.supports_temperature:
                chat_kwargs["temperature"] = temperature

            async def _chat_struct_call() -> Any:
                try:
                    return await self._client.chat.completions.create(**chat_kwargs)
                except Exception as exc:
                    raise self._translate_sdk_error(exc) from exc

            response, retries, latency_ms = await execute_with_retry(
                _chat_struct_call,
                provider=self.provider_name,
                model=self._model,
                timeout_seconds=timeout,
                max_retries=retries_cap,
            )
            choices = getattr(response, "choices", None) or []
            raw_text = getattr(choices[0].message, "content", "") if choices else ""
            try:
                parsed = response_model.model_validate_json(raw_text)
            except ValidationError as exc:
                raise ProviderResponseError(
                    f"OpenAI-compatible structured validation failed for "
                    f"{response_model.__name__}: {exc}",
                    code="validation_error",
                    provider=self.provider_name,
                    model=self._model,
                    retries_attempted=retries,
                ) from exc
            usage = self._extract_usage(response, retries, latency_ms, fallback_model=self._model)
            return StructuredGenerationResult[T](
                provider=self.provider_name,
                model=self._model,
                returned_model=usage.returned_model,
                raw_text=raw_text or parsed.model_dump_json(),
                parsed=parsed,
                usage=usage,
                raw_response_id=getattr(response, "id", None),
            )

        kwargs = self._build_common_kwargs(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        kwargs["text_format"] = response_model

        async def _call() -> Any:
            try:
                return await self._client.responses.parse(**kwargs)
            except Exception as exc:
                raise self._translate_sdk_error(exc) from exc

        response, retries, latency_ms = await execute_with_retry(
            _call,
            provider=self.provider_name,
            model=self._model,
            timeout_seconds=timeout,
            max_retries=retries_cap,
        )

        raw_text = getattr(response, "output_text", "") or ""
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            if not raw_text.strip():
                raise ProviderResponseError(
                    "OpenAI Responses API returned neither output_parsed nor output_text.",
                    code="invalid_response",
                    provider=self.provider_name,
                    model=self._model,
                    retries_attempted=retries,
                )
            try:
                parsed = response_model.model_validate_json(raw_text)
            except ValidationError as exc:
                raise ProviderResponseError(
                    f"Structured output validation failed for {response_model.__name__}: {exc}",
                    code="validation_error",
                    provider=self.provider_name,
                    model=self._model,
                    retries_attempted=retries,
                ) from exc
        elif not isinstance(parsed, response_model):
            try:
                parsed = response_model.model_validate(parsed)
            except ValidationError as exc:
                raise ProviderResponseError(
                    f"Parsed output type mismatch for {response_model.__name__}: {exc}",
                    code="validation_error",
                    provider=self.provider_name,
                    model=self._model,
                    retries_attempted=retries,
                ) from exc

        usage = self._extract_usage(response, retries, latency_ms, fallback_model=self._model)
        logger.debug(
            "OpenAI structured call succeeded (model=%s, returned=%s, retries=%d)",
            self._model,
            usage.returned_model,
            retries,
        )
        return StructuredGenerationResult[T](
            provider=self.provider_name,
            model=self._model,
            returned_model=usage.returned_model,
            raw_text=raw_text or parsed.model_dump_json(),
            parsed=parsed,
            usage=usage,
            raw_response_id=getattr(response, "id", None),
        )
