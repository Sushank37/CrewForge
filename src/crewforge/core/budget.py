"""Persistent, concurrency-safe budget manager for CrewForge.

IMPORTANT: Application-level cost estimation and pre-call reservations are
software safeguards based on configured token pricing and max-token bounds.
They are NOT a guaranteed provider-side billing cap; always configure hard
spending limits in your cloud provider dashboard as well.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from crewforge.config import CrewForgeSettings, ModelPricing
from crewforge.schemas import UsageMetrics
from crewforge.storage.database import BudgetConstraintError, Database


class BudgetError(Exception):
    """Base error for budget enforcement failures."""


class BudgetExceededError(BudgetError):
    """Raised when a model call would exceed per-run or cumulative caps."""


class UnknownModelPricingError(BudgetError):
    """Raised when a paid provider is invoked without configured pricing (fail-closed)."""


class BudgetReservation(BaseModel):
    """Active pre-call budget reservation for a single model invocation."""

    reservation_id: str = Field(default_factory=lambda: f"res_{uuid4().hex[:12]}")
    run_id: str
    provider: str
    model: str
    is_paid_provider: bool
    reserved_cost_usd: float = Field(ge=0.0)
    estimated_input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    call_units: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def estimate_token_count(text: str) -> int:
    """Conservative character-based token estimate (~3 chars/token + overhead)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3)) + 16


class BudgetManager:
    """Enforces atomic pre-call cost/call reservations and post-call reconciliation."""

    def __init__(
        self,
        database: Database,
        settings: CrewForgeSettings,
        pricing_overrides: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._pricing: dict[str, ModelPricing] = dict(settings.pricing_json)
        if pricing_overrides:
            self._pricing.update(pricing_overrides)

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Return configured pricing for a model if available."""
        return self._pricing.get(model)

    def estimate_worst_case_cost_usd(
        self,
        *,
        provider: str,
        model: str,
        is_paid_provider: bool,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> float:
        """Compute conservative pre-call USD cost reservation.

        Fails closed with UnknownModelPricingError for paid providers without
        explicit pricing configuration. Local/free providers reserve $0.00.
        """
        if not is_paid_provider:
            return 0.0

        pricing = self.get_pricing(model)
        if pricing is None:
            raise UnknownModelPricingError(
                f"Fail-closed budget protection: pricing is unknown for paid model '{model}' "
                f"(provider='{provider}'). Configure CREWFORGE_PRICING_JSON before invoking."
            )

        # Conservative 25% + 64-token margin on estimated input tokens plus full max_output_tokens.
        buffered_input = math.ceil(estimated_input_tokens * 1.25) + 64
        return pricing.estimate_cost_usd(buffered_input, max_output_tokens)

    def compute_actual_cost_usd(
        self,
        *,
        model: str,
        is_paid_provider: bool,
        usage: UsageMetrics,
    ) -> float:
        """Calculate actual cost from observed token usage and configured pricing."""
        if not is_paid_provider:
            return 0.0
        if usage.actual_cost_usd is not None:
            return usage.actual_cost_usd
        pricing = self.get_pricing(model)
        if pricing is None:
            raise UnknownModelPricingError(
                f"Cannot reconcile actual cost without pricing for paid model '{model}'."
            )
        return pricing.estimate_cost_usd(usage.input_tokens, usage.output_tokens)

    async def reserve_call(
        self,
        *,
        run_id: str,
        provider: str,
        model: str,
        is_paid_provider: bool,
        prompt_text: str,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        call_units: int = 1,
    ) -> BudgetReservation:
        """Atomically check run/cumulative caps and persist a budget reservation."""
        max_out = max_output_tokens or self._settings.max_output_tokens
        combined_input = f"{system_prompt or ''}\n{prompt_text}".strip()
        estimated_input = estimate_token_count(combined_input)

        reserved_cost = self.estimate_worst_case_cost_usd(
            provider=provider,
            model=model,
            is_paid_provider=is_paid_provider,
            estimated_input_tokens=estimated_input,
            max_output_tokens=max_out,
        )

        reservation = BudgetReservation(
            run_id=run_id,
            provider=provider,
            model=model,
            is_paid_provider=is_paid_provider,
            reserved_cost_usd=reserved_cost,
            estimated_input_tokens=estimated_input,
            max_output_tokens=max_out,
            call_units=call_units,
        )

        try:
            await self._db.reserve_budget(
                reservation_id=reservation.reservation_id,
                run_id=run_id,
                provider=provider,
                model=model,
                estimated_cost_usd=reserved_cost,
                max_model_calls_per_run=self._settings.max_model_calls_per_run,
                max_cost_per_run_usd=self._settings.max_cost_per_run_usd,
                max_cumulative_cost_usd=self._settings.max_cumulative_cost_usd,
                call_units=call_units,
            )
        except BudgetConstraintError as exc:
            raise BudgetExceededError(str(exc)) from exc

        return reservation

    async def reconcile_success(
        self,
        reservation: BudgetReservation,
        usage: UsageMetrics,
        *,
        operation: str = "generate",
    ) -> UsageMetrics:
        """Reconcile a completed call with actual token usage and cost."""
        actual_cost = self.compute_actual_cost_usd(
            model=reservation.model,
            is_paid_provider=reservation.is_paid_provider,
            usage=usage,
        )
        reconciled_usage = usage.model_copy(
            update={
                "estimated_cost_usd": reservation.reserved_cost_usd,
                "actual_cost_usd": actual_cost,
                "model_calls": max(1, usage.model_calls),
            }
        )
        await self._db.reconcile_budget(
            reservation_id=reservation.reservation_id,
            run_id=reservation.run_id,
            provider=reservation.provider,
            model=reservation.model,
            operation=operation,
            status="reconciled",
            actual_cost_usd=actual_cost,
            input_tokens=reconciled_usage.input_tokens,
            output_tokens=reconciled_usage.output_tokens,
            total_tokens=reconciled_usage.total_tokens,
            estimated_cost_usd=reservation.reserved_cost_usd,
            retries=reconciled_usage.retries,
            latency_ms=reconciled_usage.latency_ms,
            outcome="completed",
        )
        return reconciled_usage

    async def reconcile_failure(
        self,
        reservation: BudgetReservation,
        *,
        billing_known_unbilled: bool,
        retries: int = 0,
        latency_ms: float = 0.0,
        operation: str = "generate",
        reason: str = "",
    ) -> None:
        """Reconcile a failed or cancelled call conservatively.

        When billing outcome is unknown (e.g. network timeout or stream drop),
        the full reserved cost is retained against both run and cumulative caps.
        """
        if billing_known_unbilled:
            status = "released"
            actual_cost = 0.0
            outcome = "failed"
        else:
            status = "unknown"
            actual_cost = reservation.reserved_cost_usd
            outcome = "unknown_billing"

        await self._db.reconcile_budget(
            reservation_id=reservation.reservation_id,
            run_id=reservation.run_id,
            provider=reservation.provider,
            model=reservation.model,
            operation=operation,
            status=status,
            actual_cost_usd=actual_cost,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=reservation.reserved_cost_usd,
            retries=retries,
            latency_ms=latency_ms,
            outcome=outcome,
            notes=reason or None,
        )

    async def get_summary(self, run_id: str | None = None) -> dict[str, object]:
        """Return persistent budget summary for a run and cumulatively."""
        return await self._db.get_budget_summary(run_id=run_id)
