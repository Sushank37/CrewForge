"""OpenAI provider adapter using the official AsyncOpenAI Responses API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, TypeVar
from urllib.parse import urlparse

import openai
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
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
from crewforge.providers.groq_quota import (
    extract_retry_after_seconds,
    get_groq_scheduler,
    remaining_calls_var,
    run_deadline_var,
    wait_observer_var,
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
        is_free_tier: bool = False,
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
        self._is_free_tier = is_free_tier
        parsed_host = (urlparse(base_url).hostname or "").lower() if base_url else ""
        self._is_groq = parsed_host == "api.groq.com"
        self._is_groq_free_tier = self._is_groq and self._is_free_tier
        self._quota_scheduler = (
            get_groq_scheduler(base_url=base_url, model=model)
            if self._is_groq_free_tier
            else None
        )
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
        parsed_host = (
            (urlparse(settings.openai_base_url).hostname or "").lower()
            if settings.openai_base_url
            else ""
        )
        pricing = settings.pricing_json.get(settings.model)
        is_free_tier = bool(
            is_compat
            and parsed_host == "api.groq.com"
            and pricing is not None
            and pricing.input_per_1m == 0.0
            and pricing.output_per_1m == 0.0
        )
        return cls(
            model=settings.model,
            api_key=secret,
            base_url=settings.openai_base_url,
            reasoning_effort=None if is_compat else settings.openai_reasoning_effort,
            use_chat_completions=is_compat,
            is_free_tier=is_free_tier,
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
        return not self._is_free_tier

    def _effective_chat_max_tokens(self, requested: int | None) -> int:
        tokens = requested or self._default_max_output_tokens
        if self._is_groq_free_tier:
            return min(tokens, 2048)
        return tokens

    @staticmethod
    def _compact_schema_json(response_model: type[BaseModel], *, strip_descriptions: bool) -> str:
        schema = response_model.model_json_schema()
        if strip_descriptions:
            def _strip(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {
                        k: _strip(v)
                        for k, v in obj.items()
                        if k not in ("description", "title")
                    }
                if isinstance(obj, list):
                    return [_strip(item) for item in obj]
                return obj
            schema = _strip(schema)
        return json.dumps(schema, separators=(",", ":"))

    async def _create_chat_completion(
        self, chat_kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, str]]:
        completions = self._client.chat.completions
        raw_api = getattr(completions, "with_raw_response", None)
        if raw_api is not None and hasattr(raw_api, "create"):
            raw_resp = await raw_api.create(**chat_kwargs)
            headers = {str(k).lower(): str(v) for k, v in getattr(raw_resp, "headers", {}).items()}
            return raw_resp.parse(), headers
        resp = await completions.create(**chat_kwargs)
        return resp, {}

    async def _run_with_rate_limit_backoff(
        self,
        fn: Any,
        *,
        timeout_seconds: float,
        retries_cap: int,
        estimated_tokens: int = 1200,
    ) -> tuple[Any, int, float]:
        """Run provider call with quota-aware 429 scheduling (never switches models)."""
        max_rl_retries = 4 if self._is_groq_free_tier else 0
        rl_attempt = 0
        started = time.perf_counter()
        while True:
            if self._quota_scheduler is not None:
                try:
                    await self._quota_scheduler.wait_for_slot(
                        estimated_tokens,
                        deadline_monotonic=run_deadline_var.get(),
                        on_wait=wait_observer_var.get(),
                    )
                except TimeoutError as exc:
                    raise ProviderRateLimitError(
                        str(exc),
                        provider=self.provider_name,
                        model=self._model,
                        retries_attempted=rl_attempt,
                    ) from exc
            try:
                result, inner_retries, _ = await execute_with_retry(
                    fn,
                    provider=self.provider_name,
                    model=self._model,
                    timeout_seconds=timeout_seconds,
                    max_retries=retries_cap,
                )
                total_latency_ms = (time.perf_counter() - started) * 1000.0
                return result, inner_retries + rl_attempt, total_latency_ms
            except ProviderRateLimitError as rl_exc:
                rem_calls = remaining_calls_var.get()
                if rl_attempt >= max_rl_retries or (
                    rem_calls is not None and (1 + rl_attempt) >= rem_calls
                ):
                    rl_exc.retries_attempted = rl_attempt
                    raise
                if self._quota_scheduler is not None:
                    delay = self._quota_scheduler.record_rate_limit(
                        headers=rl_exc.response_headers,
                        message=str(rl_exc),
                    )
                else:
                    delay = min(15.0, 2.0 * (rl_attempt + 1))
                rl_attempt += 1
                logger.warning(
                    "Rate limited by %s (%s); quota-aware backoff %.1fs (attempt %d/%d).",
                    self.provider_name,
                    self._model,
                    delay,
                    rl_attempt,
                    max_rl_retries,
                )
                if self._quota_scheduler is None:
                    await asyncio.sleep(delay)

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
            resp_obj = getattr(exc, "response", None)
            raw_headers = getattr(resp_obj, "headers", None)
            headers = (
                {str(k).lower(): str(v) for k, v in raw_headers.items()}
                if raw_headers is not None
                else {}
            )
            msg = f"OpenAI rate limit error: {exc}"
            return ProviderRateLimitError(
                msg,
                provider=self.provider_name,
                model=self._model,
                status_code=getattr(exc, "status_code", 429),
                retry_after_seconds=extract_retry_after_seconds(headers, msg),
                response_headers=headers,
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
        resp_id = getattr(response, "id", None)
        resp_status = getattr(response, "status", None)
        inc_details = getattr(response, "incomplete_details", None)
        inc_reason = getattr(inc_details, "reason", None) if inc_details is not None else None

        usage_obj = getattr(response, "usage", None)
        total_attempts = 1 + max(0, retries)
        if usage_obj is None:
            return UsageMetrics(
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                cached_input_tokens=0,
                total_tokens=0,
                model_calls=total_attempts,
                retries=retries,
                latency_ms=latency_ms,
                usage_reported=False,
                returned_model=returned_model,
                provider_response_id=resp_id,
                response_status=resp_status,
                incomplete_reason=inc_reason,
            )

        raw_in = getattr(usage_obj, "input_tokens", None)
        if raw_in is None:
            raw_in = getattr(usage_obj, "prompt_tokens", None)
        raw_out = getattr(usage_obj, "output_tokens", None)
        if raw_out is None:
            raw_out = getattr(usage_obj, "completion_tokens", None)

        out_details = getattr(usage_obj, "output_tokens_details", None)
        if out_details is None:
            out_details = getattr(usage_obj, "completion_tokens_details", None)
        reasoning_tokens = int(getattr(out_details, "reasoning_tokens", 0) or 0)

        in_details = getattr(usage_obj, "input_tokens_details", None)
        if in_details is None:
            in_details = getattr(usage_obj, "prompt_tokens_details", None)
        cached_tokens = int(getattr(in_details, "cached_tokens", 0) or 0)

        if raw_in is None or raw_out is None:
            return UsageMetrics(
                input_tokens=int(raw_in or 0),
                output_tokens=int(raw_out or 0),
                reasoning_tokens=reasoning_tokens,
                cached_input_tokens=cached_tokens,
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
                model_calls=total_attempts,
                retries=retries,
                latency_ms=latency_ms,
                usage_reported=False,
                returned_model=returned_model,
                provider_response_id=resp_id,
                response_status=resp_status,
                incomplete_reason=inc_reason,
            )

        input_tokens = int(raw_in)
        output_tokens = int(raw_out)
        raw_total = getattr(usage_obj, "total_tokens", None)
        total_tokens = int(raw_total) if raw_total is not None else (input_tokens + output_tokens)
        return UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_tokens,
            total_tokens=max(total_tokens, input_tokens + output_tokens),
            model_calls=total_attempts,
            retries=retries,
            latency_ms=latency_ms,
            usage_reported=True,
            returned_model=returned_model,
            provider_response_id=resp_id,
            response_status=resp_status,
            incomplete_reason=inc_reason,
        )

    @staticmethod
    def _extract_refusal(response: Any) -> str | None:
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                c_type = getattr(content, "type", None)
                refusal = getattr(content, "refusal", None)
                if c_type == "refusal" or (isinstance(refusal, str) and refusal.strip()):
                    return str(refusal or "Model refused request.").strip()
        return None

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        direct = getattr(response, "output_text", None)
        if isinstance(direct, str) and direct.strip():
            return direct
        parts: list[str] = []
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) == "output_text":
                    txt = getattr(content, "text", None)
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
        return "".join(parts)

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
            eff_tokens = self._effective_chat_max_tokens(max_output_tokens)
            chat_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": eff_tokens,
            }
            if temperature is not None and self.supports_temperature:
                chat_kwargs["temperature"] = temperature

            async def _chat_call() -> tuple[Any, dict[str, str]]:
                try:
                    return await self._create_chat_completion(chat_kwargs)
                except Exception as exc:
                    raise self._translate_sdk_error(exc) from exc

            est_tokens = max(
                400,
                len(prompt) // 4 + len(system_prompt or "") // 4 + min(eff_tokens, 600),
            )
            (response, resp_headers), retries, latency_ms = await self._run_with_rate_limit_backoff(
                _chat_call,
                timeout_seconds=timeout,
                retries_cap=retries_cap,
                estimated_tokens=est_tokens,
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
            if self._quota_scheduler is not None:
                self._quota_scheduler.record_usage(usage.total_tokens, headers=resp_headers)
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
            schema_hint = self._compact_schema_json(
                response_model,
                strip_descriptions=self._is_groq_free_tier,
            )
            sys_msg = (
                system_prompt or ""
            ) + f"\nReturn strictly valid JSON matching schema: {schema_hint}"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": sys_msg.strip()},
                {"role": "user", "content": prompt},
            ]
            eff_tokens = self._effective_chat_max_tokens(max_output_tokens)
            chat_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": eff_tokens,
                "response_format": {"type": "json_object"},
            }
            if temperature is not None and self.supports_temperature:
                chat_kwargs["temperature"] = temperature

            async def _chat_struct_call() -> tuple[Any, dict[str, str]]:
                try:
                    return await self._create_chat_completion(chat_kwargs)
                except Exception as exc:
                    raise self._translate_sdk_error(exc) from exc

            est_tokens = max(
                600,
                (len(sys_msg) + len(prompt)) // 4 + min(eff_tokens, 800),
            )
            (response, resp_headers), retries, latency_ms = await self._run_with_rate_limit_backoff(
                _chat_struct_call,
                timeout_seconds=timeout,
                retries_cap=retries_cap,
                estimated_tokens=est_tokens,
            )
            choices = getattr(response, "choices", None) or []
            finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
            raw_text = getattr(choices[0].message, "content", "") if choices else ""
            usage = self._extract_usage(response, retries, latency_ms, fallback_model=self._model)
            if self._quota_scheduler is not None:
                self._quota_scheduler.record_usage(usage.total_tokens, headers=resp_headers)
            if finish_reason == "length":
                raise ProviderResponseError(
                    "OpenAI-compatible structured response truncated at max_tokens limit.",
                    code="truncated_response",
                    provider=self.provider_name,
                    model=self._model,
                    billing_outcome="billed" if usage.usage_reported else "unknown",
                    retries_attempted=retries,
                    usage=usage,
                )
            try:
                parsed = response_model.model_validate_json(raw_text)
            except ValidationError as exc:
                raise ProviderResponseError(
                    f"OpenAI-compatible structured validation failed for "
                    f"{response_model.__name__}: {exc}",
                    code="validation_error",
                    provider=self.provider_name,
                    model=self._model,
                    billing_outcome="billed" if usage.usage_reported else "unknown",
                    retries_attempted=retries,
                    usage=usage,
                ) from exc
            return StructuredGenerationResult[T](
                provider=self.provider_name,
                model=self._model,
                returned_model=usage.returned_model,
                raw_text=raw_text or parsed.model_dump_json(),
                parsed=parsed,
                usage=usage,
                raw_response_id=getattr(response, "id", None),
            )

        effective_max_out = max_output_tokens or self._default_max_output_tokens
        kwargs = self._build_common_kwargs(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=effective_max_out,
            temperature=temperature,
        )
        strict_schema = to_strict_json_schema(response_model)
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": response_model.__name__,
                "schema": strict_schema,
                "strict": True,
            }
        }

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

        # Extract usage and response metadata BEFORE attempting schema parsing.
        usage = self._extract_usage(response, retries, latency_ms, fallback_model=self._model)
        meta: dict[str, Any] = {
            "provider_response_id": usage.provider_response_id,
            "returned_model": usage.returned_model,
            "response_status": usage.response_status,
            "incomplete_reason": usage.incomplete_reason,
            "configured_max_output_tokens": effective_max_out,
        }
        billing_state = "billed" if usage.usage_reported else "unknown"

        refusal_msg = self._extract_refusal(response)
        if refusal_msg or usage.incomplete_reason == "content_filter":
            raise ProviderResponseError(
                f"OpenAI refused or filtered structured output: {refusal_msg or 'content_filter'}",
                code="refusal",
                provider=self.provider_name,
                model=self._model,
                billing_outcome=billing_state,
                retries_attempted=retries,
                usage=usage,
                response_metadata={**meta, "validation_outcome": "refusal"},
            )

        if usage.response_status == "incomplete":
            reason = usage.incomplete_reason or "incomplete"
            raise ProviderResponseError(
                f"OpenAI structured response incomplete (reason={reason}, "
                f"max_output_tokens={effective_max_out}).",
                code="truncated_response"
                if reason == "max_output_tokens"
                else "incomplete_response",
                provider=self.provider_name,
                model=self._model,
                billing_outcome=billing_state,
                retries_attempted=retries,
                usage=usage,
                response_metadata={**meta, "validation_outcome": f"incomplete:{reason}"},
            )

        raw_text = self._extract_output_text(response)
        if not raw_text.strip():
            raise ProviderResponseError(
                "OpenAI Responses API returned empty structured output_text.",
                code="invalid_response",
                provider=self.provider_name,
                model=self._model,
                billing_outcome=billing_state,
                retries_attempted=retries,
                usage=usage,
                response_metadata={**meta, "validation_outcome": "empty_output"},
            )

        try:
            parsed = response_model.model_validate_json(raw_text)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"Structured output validation failed for {response_model.__name__}: {exc}",
                code="validation_error",
                provider=self.provider_name,
                model=self._model,
                billing_outcome=billing_state,
                retries_attempted=retries,
                usage=usage,
                response_metadata={**meta, "validation_outcome": "schema_validation_failed"},
            ) from exc

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
            raw_response_id=usage.provider_response_id,
        )
