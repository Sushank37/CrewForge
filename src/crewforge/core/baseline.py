"""Single-agent baseline execution runner powered by ModelExecutor."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from crewforge.config import CrewForgeSettings, ProviderType, sanitize_error_message
from crewforge.core.budget import BudgetError, BudgetExceededError, BudgetManager
from crewforge.core.executor import ModelExecutor, RunDeadlineExceededError
from crewforge.providers.base import ModelProvider, ProviderError
from crewforge.providers.ollama_provider import OllamaProvider
from crewforge.providers.openai_provider import OpenAIProvider
from crewforge.schemas import ExecutionEvent, RunResult, RunStatus
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
    """Executes single-agent baseline runs via the shared ModelExecutor."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        database: Database,
        budget_manager: BudgetManager,
        settings: CrewForgeSettings,
        executor: ModelExecutor | None = None,
    ) -> None:
        self._provider = provider
        self._db = database
        self._budget = budget_manager
        self._settings = settings
        self._executor = executor or ModelExecutor(
            provider=provider,
            database=database,
            budget_manager=budget_manager,
            settings=settings,
        )

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

        try:
            response = await self._executor.execute_structured(
                run_id=resolved_run_id,
                actor="baseline",
                operation="baseline",
                prompt=cleaned_prompt,
                response_model=BaselineModelOutput,
                system_prompt=BASELINE_SYSTEM_PROMPT,
                max_output_tokens=self._settings.max_output_tokens,
                timeout_seconds=self._settings.request_timeout_seconds,
            )
            total_latency_ms = (time.perf_counter() - start_perf) * 1000.0
            aggregated_usage = await self._db.aggregate_run_usage(resolved_run_id)

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
                usage=aggregated_usage,
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
            aggregated_usage = await self._db.aggregate_run_usage(resolved_run_id)
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
                usage=aggregated_usage,
                latency_ms=elapsed_ms,
                error="Run cancelled.",
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(cancelled_result)
            raise

        except BudgetError as exc:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            safe_err = sanitize_error_message(str(exc))
            aggregated_usage = await self._db.aggregate_run_usage(resolved_run_id)
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="budget_rejected",
                    actor="budget_manager",
                    message=safe_err,
                )
            )
            status = (
                RunStatus.BUDGET_EXCEEDED
                if isinstance(exc, BudgetExceededError)
                else RunStatus.FAILED
            )
            failed_result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=status,
                prompt=cleaned_prompt,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                usage=aggregated_usage,
                latency_ms=elapsed_ms,
                error=safe_err,
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(failed_result)
            return failed_result

        except (ProviderError, RunDeadlineExceededError) as exc:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            safe_err = sanitize_error_message(str(exc))
            aggregated_usage = await self._db.aggregate_run_usage(resolved_run_id)
            failed_result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=RunStatus.FAILED,
                prompt=cleaned_prompt,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                usage=aggregated_usage,
                latency_ms=elapsed_ms,
                error=safe_err,
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(failed_result)
            return failed_result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            safe_err = sanitize_error_message(
                f"Unexpected baseline error ({type(exc).__name__}): {exc}"
            )
            aggregated_usage = await self._db.aggregate_run_usage(resolved_run_id)
            await self._db.append_event(
                ExecutionEvent(
                    run_id=resolved_run_id,
                    event_type="run_failed",
                    actor="baseline",
                    message=safe_err,
                )
            )
            failed_result = RunResult(
                run_id=resolved_run_id,
                mode="baseline",
                status=RunStatus.FAILED,
                prompt=cleaned_prompt,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                usage=aggregated_usage,
                latency_ms=elapsed_ms,
                error=safe_err,
                created_at=created_at,
                completed_at=datetime.now(UTC),
            )
            await self._db.complete_run(failed_result)
            return failed_result
