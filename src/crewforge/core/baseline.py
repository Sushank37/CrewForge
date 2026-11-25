"""Single-agent baseline execution runner."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.core.budget import BudgetError, BudgetManager, BudgetReservation
from crewforge.providers.base import ModelProvider, ProviderError
from crewforge.providers.ollama_provider import OllamaProvider
from crewforge.providers.openai_provider import OpenAIProvider
from crewforge.schemas import ExecutionEvent, RunResult, RunStatus, UsageMetrics
from crewforge.storage.database import Database

BASELINE_SYSTEM_PROMPT = (
    "You are a careful, thorough single-agent problem solver. "
    "Provide a direct, well-reasoned, and genuinely helpful answer to the user's prompt. "
    "Do not invent facts or hide uncertainty: list any important missing information in "
    "`missing_information`, state explicit working assumptions in `assumptions`, and summarize "
    "overall certainty or caveats in `confidence_notes`."
)


class BaselineModelOutput(BaseModel):
    """Structured schema expected from the baseline model invocation."""

    answer: str = Field(min_length=1, description="Direct, useful answer to the prompt.")
    missing_information: list[str] = Field(
        default_factory=list,
        description="Important missing data, ambiguities, or unverified dependencies.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions made while answering.",
    )
    confidence_notes: str | None = Field(
        default=None,
        description="Brief notes on confidence, uncertainty, or scope limitations.",
    )


def build_provider_from_settings(settings: CrewForgeSettings) -> ModelProvider:
    """Instantiate the configured production provider without silent fallbacks."""
    if settings.provider == ProviderType.OPENAI:
        return OpenAIProvider.from_settings(settings)
    if settings.provider == ProviderType.OLLAMA:
        return OllamaProvider.from_settings(settings)
    raise ValueError(f"Unsupported provider: {settings.provider}")


class BaselineRunner:
    """Executes single-agent baseline runs with budget checks and trace persistence."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        database: Database,
        budget_manager: BudgetManager,
        settings: CrewForgeSettings,
    ) -> None:
        self._provider = provider
        self._db = database
        self._budget = budget_manager
        self._settings = settings

    async def run(self, prompt: str, *, run_id: str | None = None) -> RunResult:
        """Execute a baseline run for an arbitrary prompt and persist full trace."""
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt must be a non-empty string.")

        resolved_run_id = run_id or f"run_{uuid4().hex[:12]}"
        created_at = datetime.now(UTC)
        start_perf = time.perf_counter()

        run_config = self._settings.to_run_config().model_copy(
            update={
                "provider": self._provider.provider_name,
                "model": self._provider.model_name,
            }
        )
        await self._db.create_run(
            run_id=resolved_run_id,
            mode="baseline",
            prompt=cleaned_prompt,
            config=run_config,
        )
        await self._db.append_event(
            ExecutionEvent(
                run_id=resolved_run_id,
                event_type="run_started",
                actor="baseline",
                message="Started single-agent baseline run.",
                payload={
                    "provider": self._provider.provider_name,
                    "model": self._provider.model_name,
                },
            )
        )

        reservation: BudgetReservation | None = None
        try:
            reservation = await self._budget.reserve_call(
                run_id=resolved_run_id,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                is_paid_provider=self._provider.is_paid_provider,
                prompt_text=cleaned_prompt,
                system_prompt=BASELINE_SYSTEM_PROMPT,
                max_output_tokens=self._settings.max_output_tokens,
            )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="budget_reserved",
                    actor="budget_manager",
                    message="Reserved budget for baseline model invocation.",
                    payload={
                        "reservation_id": reservation.reservation_id,
                        "reserved_cost_usd": reservation.reserved_cost_usd,
                        "max_output_tokens": reservation.max_output_tokens,
                    },
                )
            )

            response = await self._provider.generate_structured(
                cleaned_prompt,
                BaselineModelOutput,
                system_prompt=BASELINE_SYSTEM_PROMPT,
                max_output_tokens=self._settings.max_output_tokens,
                timeout_seconds=self._settings.request_timeout_seconds,
            )
            reconciled_usage = await self._budget.reconcile_success(
                reservation,
                response.usage,
                operation="baseline",
            )
            total_latency_ms = (time.perf_counter() - start_perf) * 1000.0
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="model_call_completed",
                    actor="baseline",
                    message="Baseline model call completed and reconciled.",
                    payload={
                        "provider": response.provider,
                        "model": response.model,
                        "input_tokens": reconciled_usage.input_tokens,
                        "output_tokens": reconciled_usage.output_tokens,
                        "total_tokens": reconciled_usage.total_tokens,
                        "actual_cost_usd": reconciled_usage.actual_cost_usd,
                        "retries": reconciled_usage.retries,
                        "latency_ms": reconciled_usage.latency_ms,
                    },
                )
            )

            result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=RunStatus.COMPLETED,
                prompt=cleaned_prompt,
                provider=response.provider,
                model=response.model,
                answer=response.parsed.answer,
                missing_information=response.parsed.missing_information,
                assumptions=response.parsed.assumptions,
                confidence_notes=response.parsed.confidence_notes,
                usage=reconciled_usage,
                latency_ms=total_latency_ms,
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="run_completed",
                    actor="baseline",
                    message="Baseline run completed successfully.",
                    payload={"latency_ms": total_latency_ms},
                )
            )
            await self._db.complete_run(result)
            return result

        except asyncio.CancelledError:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            if reservation is not None:
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=False,
                    latency_ms=elapsed_ms,
                    operation="baseline",
                    reason="Run cancelled while model call was in flight.",
                )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="run_cancelled",
                    actor="baseline",
                    message="Baseline run was cancelled.",
                )
            )
            cancelled_result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=RunStatus.CANCELLED,
                prompt=cleaned_prompt,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                latency_ms=elapsed_ms,
                error="Run cancelled.",
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(cancelled_result)
            raise

        except BudgetError as exc:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="budget_rejected",
                    actor="budget_manager",
                    message=str(exc),
                )
            )
            status = (
                RunStatus.BUDGET_EXCEEDED
                if exc.__class__.__name__ == "BudgetExceededError"
                else RunStatus.FAILED
            )
            failed_result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=status,
                prompt=cleaned_prompt,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                latency_ms=elapsed_ms,
                error=str(exc),
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(failed_result)
            return failed_result

        except ProviderError as exc:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            if reservation is not None:
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=(exc.billing_outcome == "unbilled"),
                    retries=exc.retries_attempted,
                    latency_ms=elapsed_ms,
                    operation="baseline",
                    reason=exc.message,
                )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="model_call_failed",
                    actor="baseline",
                    message=exc.message,
                    payload=exc.to_dict(),
                )
            )
            failed_result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=RunStatus.FAILED,
                prompt=cleaned_prompt,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                usage=UsageMetrics(
                    estimated_cost_usd=(
                        reservation.reserved_cost_usd if reservation is not None else 0.0
                    ),
                    actual_cost_usd=(
                        0.0
                        if reservation is None or exc.billing_outcome == "unbilled"
                        else reservation.reserved_cost_usd
                    ),
                    model_calls=1 if reservation is not None else 0,
                    retries=exc.retries_attempted,
                    latency_ms=elapsed_ms,
                ),
                latency_ms=elapsed_ms,
                error=exc.message,
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(failed_result)
            return failed_result
