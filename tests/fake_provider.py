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


def checklist_fixture(handler):
    """Translate pre-checklist scripted responses into the current test wire format.

    In these legacy fixtures each scripted requirement is a single test obligation.
    Their scripted approvals and blocking decisions still control the review outcome.
    New coverage and planner-validation tests should script the actual contract directly.
    """
    import ast
    import re

    def respond(prompt, schema):
        result = handler(prompt, schema)
        name = schema.__name__ if schema else ""
        if name in {"PlannerOutput", "ExecutionPlan"}:
            if isinstance(result, BaseModel):
                result = result.model_dump()
            if not isinstance(result, dict) or result.get("deliverables") is not None:
                return result
            tasks = result.get("tasks") or []
            if not tasks:
                return result
            return result | {"deliverables": [
                {
                    "deliverable_id": f"deliv_{idx}", "requirement_ref": f"req_{idx}",
                    "text": req, "kind": (
                        "recommendation" if "recommend" in req.lower() else "explanation"
                    ),
                    "task_ref": tasks[min(idx - 1, len(tasks) - 1)]["task_ref"],
                }
                for idx, req in enumerate(result.get("user_requirements", []), start=1)
            ]}
        if not isinstance(result, dict):
            return result
        if name in {"SpecialistOutput", "WorkerOutput", "FollowUpOutput"}:
            requirements = {
                req.strip().lower(): f"req_{idx}"
                for idx, req in re.findall(r"^- \[req_(\d+)\] (.+)$", prompt, re.M)
            }
            if not requirements:
                return result
            for key in ("claims", "new_claims"):
                for claim in result.get(key, []):
                    refs = list(claim.get("requirement_refs") or [])
                    if any(ref.startswith("deliv_") for ref in refs):
                        continue
                    mapped = [requirements.get(ref.strip().lower(), ref) for ref in refs]
                    if not mapped and len(requirements) == 1:
                        mapped = list(requirements.values())
                    claim["requirement_refs"] = list(dict.fromkeys([
                        *refs, *(ref.replace("req_", "deliv_", 1) for ref in mapped
                                 if re.fullmatch(r"req_\d+", ref)),
                    ]))
        if name in {"ReviewerBatchOutput", "ReviewBatchOutput"}:
            claims = {}
            for cid, version, alias, refs in re.findall(
                r"Claim (\w+) \(version=(\d+), alias=(\w+),[^\n]*\n"
                r"  Statement: [^\n]*\n  Requested coverage: (\[[^\n]*\])", prompt,
            ):
                covered = [ref for ref in ast.literal_eval(refs) if ref.startswith("deliv_")]
                claims[cid, int(version)] = covered
                claims[cid, None] = covered
                claims[alias, None] = covered
            for verdict in result.get("verdicts", []):
                key = (verdict.get("claim_id"), verdict.get("claim_version"))
                if (
                    verdict.get("decision") == "approve"
                    and "covered_deliverable_refs" not in verdict
                ):
                    verdict["covered_deliverable_refs"] = claims.get(key, [])
        return result

    return respond


class ChecklistFixtureProvider(FakeProvider):
    """Adapt legacy orchestration fixtures while exercising the new production planner."""

    def __init__(self, *, handler=None, **kwargs):
        super().__init__(handler=checklist_fixture(handler) if handler else None, **kwargs)
