"""Shared budget-aware ModelExecutor with per-attempt reservation and bounded semaphore."""

from __future__ import annotations

import asyncio
import time
from typing import TypeVar

from pydantic import BaseModel

from crewforge.config import CrewForgeSettings, sanitize_error_message
from crewforge.core.budget import BudgetExceededError, BudgetManager, BudgetReservation
from crewforge.providers.base import (
    ModelProvider,
    ProviderError,
    ProviderTimeoutError,
    StructuredGenerationResult,
    TextGenerationResult,
)
from crewforge.schemas import ExecutionEvent
from crewforge.storage.database import Database

T = TypeVar("T", bound=BaseModel)


class RunDeadlineExceededError(Exception):
    """Raised when the overall run deadline elapses before or during a model call."""


class ModelExecutor:
    """Executes all model attempts through a shared semaphore and per-attempt reservations."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        database: Database,
        budget_manager: BudgetManager,
        settings: CrewForgeSettings,
        base_backoff_seconds: float = 0.25,
    ) -> None:
        self._provider = provider
        self._db = database
        self._budget = budget_manager
        self._settings = settings
        self._base_backoff_seconds = base_backoff_seconds
        self._semaphore = asyncio.Semaphore(settings.effective_max_concurrency)

    @property
    def provider(self) -> ModelProvider:
        return self._provider

    @property
    def concurrency_limit(self) -> int:
        return self._settings.effective_max_concurrency

    def _effective_timeout(
        self,
        requested_timeout: float | None,
        run_deadline_monotonic: float | None,
    ) -> float:
        timeout = requested_timeout or self._settings.request_timeout_seconds
        if run_deadline_monotonic is None:
            return timeout
        remaining = run_deadline_monotonic - time.monotonic()
        if remaining <= 0.0:
            raise RunDeadlineExceededError("Run deadline exceeded before model attempt dispatch.")
        return min(timeout, remaining)

    async def execute_structured(
        self,
        *,
        run_id: str,
        actor: str,
        operation: str,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        reserve_calls_after: int = 0,
        run_deadline_monotonic: float | None = None,
    ) -> StructuredGenerationResult[T]:
        """Run a structured generation call with per-attempt reservation and bounded retries."""
        requested_out = max_output_tokens or self._settings.output_limit_for_operation(operation)
        bounded_out = min(requested_out, self._settings.max_output_tokens)
        max_retries = self._settings.max_retries

        attempt_index = 0
        while True:
            if reserve_calls_after:
                usage = await self._db.get_budget_summary(run_id)
                remaining = (
                    self._settings.max_model_calls_per_run - int(usage["run_committed_calls"])
                )
                if remaining <= reserve_calls_after:
                    raise BudgetExceededError("Remaining calls reserved for review or synthesis.")
            attempt_timeout = self._effective_timeout(timeout_seconds, run_deadline_monotonic)
            reservation: BudgetReservation = await self._budget.reserve_call(
                run_id=run_id,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                is_paid_provider=self._provider.is_paid_provider,
                prompt_text=prompt,
                system_prompt=system_prompt,
                response_model=response_model,
                max_output_tokens=bounded_out,
                allocation_category=self._settings.budget_allocation_category.value,
            )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="budget_reserved",
                    actor=actor,
                    message=f"Reserved budget for {operation} attempt #{attempt_index + 1}.",
                    payload={
                        "operation": operation,
                        "attempt": attempt_index + 1,
                        "reservation_id": reservation.reservation_id,
                        "reserved_cost_usd": reservation.reserved_cost_usd,
                        "estimated_input_tokens": reservation.estimated_input_tokens,
                        "max_output_tokens": reservation.max_output_tokens,
                        "effective_max_output_tokens": bounded_out,
                    },
                )
            )

            attempt_start = time.perf_counter()
            acquired = False
            try:
                wait_budget = self._effective_timeout(timeout_seconds, run_deadline_monotonic)
                try:
                    await asyncio.wait_for(self._semaphore.acquire(), timeout=wait_budget)
                    acquired = True
                except TimeoutError as exc:
                    if (
                        run_deadline_monotonic is not None
                        and time.monotonic() >= run_deadline_monotonic
                    ):
                        raise RunDeadlineExceededError(
                            "Run deadline expired while waiting for concurrency slot."
                        ) from exc
                    raise TimeoutError("Timed out waiting for concurrency slot.") from exc

                attempt_timeout = self._effective_timeout(timeout_seconds, run_deadline_monotonic)
                response = await asyncio.wait_for(
                    self._provider.generate_structured(
                        prompt,
                        response_model,
                        system_prompt=system_prompt,
                        max_output_tokens=bounded_out,
                        temperature=temperature,
                        timeout_seconds=attempt_timeout,
                        max_retries=0,
                    ),
                    timeout=attempt_timeout,
                )

                attempt_latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                total_retries = attempt_index + max(0, int(response.usage.retries))
                total_calls = max(1 + total_retries, int(response.usage.model_calls))
                usage_with_attempt = response.usage.model_copy(
                    update={
                        "retries": total_retries,
                        "model_calls": total_calls,
                        "latency_ms": attempt_latency_ms,
                    }
                )
                reconciled_usage = await self._budget.reconcile_success(
                    reservation,
                    usage_with_attempt,
                    operation=operation,
                )
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="model_call_completed",
                        actor=actor,
                        message=f"Completed {operation} attempt #{attempt_index + 1}.",
                        payload={
                            "operation": operation,
                            "attempt": attempt_index + 1,
                            "provider": response.provider,
                            "model": response.model,
                            "returned_model": response.returned_model,
                            "input_tokens": reconciled_usage.input_tokens,
                            "output_tokens": reconciled_usage.output_tokens,
                            "reasoning_tokens": reconciled_usage.reasoning_tokens,
                            "cached_input_tokens": reconciled_usage.cached_input_tokens,
                            "total_tokens": reconciled_usage.total_tokens,
                            "actual_cost_usd": reconciled_usage.actual_cost_usd,
                            "usage_reported": reconciled_usage.usage_reported,
                            "provider_response_id": reconciled_usage.provider_response_id,
                            "response_status": reconciled_usage.response_status,
                            "effective_max_output_tokens": bounded_out,
                            "latency_ms": reconciled_usage.latency_ms,
                        },
                    )
                )
                return response.model_copy(update={"usage": reconciled_usage})

            except asyncio.CancelledError:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=not acquired,
                    retries=attempt_index,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason="Attempt cancelled while waiting or in flight.",
                )
                raise

            except TimeoutError as exc:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                err = ProviderTimeoutError(
                    f"Request to {self._provider.provider_name}/{self._provider.model_name} "
                    f"timed out after {attempt_timeout:.2f}s.",
                    provider=self._provider.provider_name,
                    model=self._provider.model_name,
                    retries_attempted=attempt_index,
                )
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=not acquired,
                    retries=attempt_index,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason=err.message,
                )
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="model_call_failed",
                        actor=actor,
                        message=err.message,
                        payload={
                            **err.to_dict(),
                            "operation": operation,
                            "attempt": attempt_index + 1,
                        },
                    )
                )
                if (
                    run_deadline_monotonic is not None
                    and time.monotonic() >= run_deadline_monotonic
                ):
                    raise RunDeadlineExceededError("Run deadline elapsed after timeout.") from exc
                if attempt_index >= max_retries:
                    raise err from exc

            except RunDeadlineExceededError:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=True,
                    retries=attempt_index,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason="Run deadline expired before provider dispatch.",
                )
                raise

            except ProviderError as err:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                total_retries = attempt_index + max(0, int(err.retries_attempted))
                err.retries_attempted = total_retries
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=(err.billing_outcome == "unbilled"),
                    usage=err.usage,
                    retries=total_retries,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason=err.message,
                )
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="model_call_failed",
                        actor=actor,
                        message=err.message,
                        payload={
                            **err.to_dict(),
                            "operation": operation,
                            "attempt": attempt_index + 1,
                            "effective_max_output_tokens": bounded_out,
                        },
                    )
                )
                if not err.retryable or attempt_index >= max_retries:
                    raise

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                safe_msg = sanitize_error_message(
                    f"Unexpected {type(exc).__name__} during {operation}: {exc}"
                )
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=False,
                    retries=attempt_index,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason=safe_msg,
                )
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="model_call_failed",
                        actor=actor,
                        message=safe_msg,
                        payload={"operation": operation, "attempt": attempt_index + 1},
                    )
                )
                raise
            finally:
                if acquired:
                    self._semaphore.release()

            attempt_index += 1
            delay = self._base_backoff_seconds * (2 ** (attempt_index - 1))
            if run_deadline_monotonic is not None:
                remaining = run_deadline_monotonic - time.monotonic()
                if remaining <= delay:
                    raise RunDeadlineExceededError("Insufficient run deadline remaining for retry.")
            await asyncio.sleep(delay)

    async def execute_text(
        self,
        *,
        run_id: str,
        actor: str,
        operation: str,
        prompt: str,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        run_deadline_monotonic: float | None = None,
    ) -> TextGenerationResult:
        """Run an unstructured text generation call with per-attempt reservation."""
        requested_out = max_output_tokens or self._settings.output_limit_for_operation(operation)
        bounded_out = min(requested_out, self._settings.max_output_tokens)
        max_retries = self._settings.max_retries

        attempt_index = 0
        while True:
            attempt_timeout = self._effective_timeout(timeout_seconds, run_deadline_monotonic)
            reservation = await self._budget.reserve_call(
                run_id=run_id,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
                is_paid_provider=self._provider.is_paid_provider,
                prompt_text=prompt,
                system_prompt=system_prompt,
                max_output_tokens=bounded_out,
                allocation_category=self._settings.budget_allocation_category.value,
            )
            attempt_start = time.perf_counter()
            acquired = False
            try:
                wait_budget = self._effective_timeout(timeout_seconds, run_deadline_monotonic)
                await asyncio.wait_for(self._semaphore.acquire(), timeout=wait_budget)
                acquired = True
                attempt_timeout = self._effective_timeout(timeout_seconds, run_deadline_monotonic)
                response = await asyncio.wait_for(
                    self._provider.generate_text(
                        prompt,
                        system_prompt=system_prompt,
                        max_output_tokens=bounded_out,
                        temperature=temperature,
                        timeout_seconds=attempt_timeout,
                        max_retries=0,
                    ),
                    timeout=attempt_timeout,
                )
                attempt_latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                total_retries = attempt_index + max(0, int(response.usage.retries))
                total_calls = max(1 + total_retries, int(response.usage.model_calls))
                reconciled_usage = await self._budget.reconcile_success(
                    reservation,
                    response.usage.model_copy(
                        update={
                            "retries": total_retries,
                            "model_calls": total_calls,
                            "latency_ms": attempt_latency_ms,
                        }
                    ),
                    operation=operation,
                )
                return response.model_copy(update={"usage": reconciled_usage})
            except asyncio.CancelledError:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=not acquired,
                    retries=attempt_index,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason="Attempt cancelled while waiting or in flight.",
                )
                raise
            except TimeoutError as exc:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                err = ProviderTimeoutError(
                    f"Request to {self._provider.provider_name}/{self._provider.model_name} "
                    f"timed out after {attempt_timeout:.2f}s.",
                    provider=self._provider.provider_name,
                    model=self._provider.model_name,
                    retries_attempted=attempt_index,
                )
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=not acquired,
                    retries=attempt_index,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason=err.message,
                )
                if attempt_index >= max_retries:
                    raise err from exc
            except ProviderError as err:
                elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
                total_retries = attempt_index + max(0, int(err.retries_attempted))
                err.retries_attempted = total_retries
                await self._budget.reconcile_failure(
                    reservation,
                    billing_known_unbilled=(err.billing_outcome == "unbilled"),
                    usage=err.usage,
                    retries=total_retries,
                    latency_ms=elapsed_ms,
                    operation=operation,
                    reason=err.message,
                )
                if not err.retryable or attempt_index >= max_retries:
                    raise
            finally:
                if acquired:
                    self._semaphore.release()
            attempt_index += 1
            await asyncio.sleep(self._base_backoff_seconds * (2 ** (attempt_index - 1)))
