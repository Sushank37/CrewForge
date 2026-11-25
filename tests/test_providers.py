"""Offline tests for model providers, structured validation, retries, and timeouts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from crewforge.providers.base import (
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from crewforge.providers.ollama_provider import OllamaProvider
from crewforge.providers.openai_provider import OpenAIProvider
from tests.fake_provider import FakeProvider


class SampleAnswer(BaseModel):
    answer: str
    score: int


async def test_fake_provider_retries_then_succeeds() -> None:
    attempts = 0

    def handler(_prompt: str, _model: type[BaseModel] | None) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderRateLimitError("rate limited", provider="fake", model="fake-model-v1")
        return {"answer": "ok", "score": 7}

    provider = FakeProvider(max_retries=2, handler=handler)
    res = await provider.generate_structured("test", SampleAnswer)
    assert res.parsed == SampleAnswer(answer="ok", score=7)
    assert res.usage.retries == 1
    assert res.usage.total_tokens == 66


async def test_fake_provider_timeout_and_invalid_structured_response() -> None:
    slow_provider = FakeProvider(delay_seconds=0.2, timeout_seconds=0.02, max_retries=0)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        await slow_provider.generate_structured("hello", SampleAnswer)
    assert exc_info.value.billing_outcome == "unknown"

    bad_provider = FakeProvider(handler=lambda _p, _m: "{not valid json")
    with pytest.raises(ProviderResponseError) as bad_info:
        await bad_provider.generate_structured("hello", SampleAnswer)
    assert bad_info.value.code == "validation_error"


async def test_provider_cancellation_propagates() -> None:
    slow_provider = FakeProvider(delay_seconds=1.0, timeout_seconds=5.0, max_retries=0)
    task = asyncio.create_task(slow_provider.generate_text("cancel me"))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_openai_provider_requires_key_and_uses_responses_api() -> None:
    with pytest.raises(ProviderConfigurationError):
        OpenAIProvider(model="test-openai-model", api_key="")

    captured: list[dict[str, Any]] = []

    class FakeResponsesAPI:
        async def create(self, **kwargs: Any) -> Any:
            captured.append({"method": "create", "kwargs": kwargs})
            if "text" in kwargs:
                if len(captured) == 2:
                    raise openai.APITimeoutError(
                        request=httpx.Request("POST", "https://api.openai.com/v1/responses")
                    )
                return SimpleNamespace(
                    id="resp_parse_1",
                    model="test-openai-model",
                    status="completed",
                    incomplete_details=None,
                    output_text='{"answer":"parsed","score":9}',
                    usage=SimpleNamespace(
                        input_tokens=20,
                        output_tokens=8,
                        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                        input_tokens_details=SimpleNamespace(cached_tokens=5),
                        total_tokens=28,
                    ),
                )
            return SimpleNamespace(
                id="resp_text_1",
                model="test-openai-model",
                status="completed",
                output_text="hello world",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            )

    fake_client = SimpleNamespace(responses=FakeResponsesAPI())
    provider = OpenAIProvider(
        model="test-openai-model",
        api_key="test-key",
        max_retries=1,
        client=fake_client,  # type: ignore[arg-type]
    )

    text_res = await provider.generate_text("say hi", system_prompt="be concise")
    assert text_res.text == "hello world"
    assert captured[0]["method"] == "create"
    assert captured[0]["kwargs"]["instructions"] == "be concise"

    struct_res = await provider.generate_structured("return json", SampleAnswer)
    assert struct_res.parsed == SampleAnswer(answer="parsed", score=9)
    assert struct_res.usage.retries == 1
    assert struct_res.usage.reasoning_tokens == 3
    assert struct_res.usage.cached_input_tokens == 5
    # Reasoning tokens (3) are already included in output_tokens (8) -> total is 28, not 31
    assert struct_res.usage.total_tokens == 28
    fmt = captured[1]["kwargs"]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True


async def test_openai_provider_invalid_structured_output() -> None:
    class BadResponsesAPI:
        async def create(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                id="resp_bad",
                model="test-openai-model",
                status="completed",
                incomplete_details=None,
                output_text='{"answer": "missing score"}',
                usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
            )

    provider = OpenAIProvider(
        model="test-openai-model",
        api_key="test-key",
        max_retries=0,
        client=SimpleNamespace(responses=BadResponsesAPI()),  # type: ignore[arg-type]
    )
    with pytest.raises(ProviderResponseError, match="validation failed") as exc_info:
        await provider.generate_structured("bad", SampleAnswer)
    assert exc_info.value.usage is not None
    assert exc_info.value.usage.input_tokens == 10
    assert exc_info.value.usage.output_tokens == 4
    assert exc_info.value.billing_outcome == "billed"


async def test_openai_provider_incomplete_refusal_and_missing_usage() -> None:
    """Verify responses.create preserves usage on incomplete, refusal, and missing-usage."""

    class IncompleteAPI:
        async def create(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                id="resp_inc",
                model="gpt-6-luna",
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                output_text='{"answer": "trunc',
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=2560,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=1800),
                    total_tokens=2660,
                ),
            )

    p_inc = OpenAIProvider(
        model="gpt-6-luna",
        api_key="test-key",
        max_retries=0,
        client=SimpleNamespace(responses=IncompleteAPI()),  # type: ignore[arg-type]
    )
    with pytest.raises(ProviderResponseError, match="max_output_tokens") as e_inc:
        await p_inc.generate_structured("q", SampleAnswer, max_output_tokens=2560)
    assert e_inc.value.code == "truncated_response"
    assert e_inc.value.billing_outcome == "billed"
    assert e_inc.value.usage is not None
    assert e_inc.value.usage.output_tokens == 2560
    assert e_inc.value.usage.reasoning_tokens == 1800
    assert e_inc.value.usage.total_tokens == 2660

    class RefusalAPI:
        async def create(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                id="resp_ref",
                model="gpt-6-luna",
                status="completed",
                incomplete_details=None,
                output=[
                    SimpleNamespace(
                        content=[SimpleNamespace(type="refusal", refusal="Policy refusal.")]
                    )
                ],
                output_text="",
                usage=SimpleNamespace(input_tokens=50, output_tokens=12, total_tokens=62),
            )

    p_ref = OpenAIProvider(
        model="gpt-6-luna",
        api_key="test-key",
        max_retries=0,
        client=SimpleNamespace(responses=RefusalAPI()),  # type: ignore[arg-type]
    )
    with pytest.raises(ProviderResponseError, match="refused") as e_ref:
        await p_ref.generate_structured("q", SampleAnswer)
    assert e_ref.value.code == "refusal"
    assert e_ref.value.billing_outcome == "billed"
    assert e_ref.value.usage is not None
    assert e_ref.value.usage.output_tokens == 12

    class NoUsageAPI:
        async def create(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                id="resp_nousage",
                model="gpt-6-luna",
                status="completed",
                incomplete_details=None,
                output_text='{"answer": "bad"}',
                usage=None,
            )

    p_no = OpenAIProvider(
        model="gpt-6-luna",
        api_key="test-key",
        max_retries=0,
        client=SimpleNamespace(responses=NoUsageAPI()),  # type: ignore[arg-type]
    )
    with pytest.raises(ProviderResponseError) as e_no:
        await p_no.generate_structured("q", SampleAnswer)
    assert e_no.value.billing_outcome == "unknown"


async def test_ollama_provider_chat_structured_and_errors() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        if body["messages"][-1]["content"] == "bad-json":
            return httpx.Response(200, json={"message": {"content": "not-json"}})
        if body["messages"][-1]["content"] == "server-500":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"answer":"local","score":5}'},
                "prompt_eval_count": 30,
                "eval_count": 12,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=transport) as client:
        provider = OllamaProvider(
            model="qwen3:4b",
            max_retries=0,
            num_ctx=4096,
            think=False,
            client=client,
        )
        res = await provider.generate_structured("good", SampleAnswer, system_prompt="sys")
        assert res.parsed == SampleAnswer(answer="local", score=5)
        assert res.usage.input_tokens == 30
        assert res.usage.output_tokens == 12
        assert seen_payloads[0]["format"] == SampleAnswer.model_json_schema()
        assert seen_payloads[0]["think"] is False
        assert seen_payloads[0]["options"]["num_ctx"] == 4096

        with pytest.raises(ProviderResponseError):
            await provider.generate_structured("bad-json", SampleAnswer)

        with pytest.raises(ProviderError) as server_err:
            await provider.generate_structured("server-500", SampleAnswer)
        assert server_err.value.status_code == 503
        assert server_err.value.retryable is True


async def test_ollama_provider_connection_error() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(failing_handler)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=transport) as client:
        provider = OllamaProvider(model="qwen3:4b", max_retries=0, client=client)
        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.generate_text("ping")
        assert exc_info.value.billing_outcome == "unbilled"
