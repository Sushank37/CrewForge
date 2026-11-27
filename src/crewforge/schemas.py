"""Typed Pydantic v2 data contracts for CrewForge (Phases 1-4)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ClaimStatus(StrEnum):
    """Lifecycle verification state of a claim in the ledger."""

    ASSERTED = "asserted"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"
    STALE = "stale"


class ClaimCategory(StrEnum):
    """Domain-agnostic epistemic category of a claim."""

    GIVEN_FACT = "given_fact"
    EXTERNAL_FACT = "external_factual_assertion"
    COMPUTATION = "computation"
    INFERENCE = "logical_inference"
    ASSUMPTION = "assumption"
    RECOMMENDATION = "recommendation"
    UNCERTAINTY = "uncertainty"


class VerificationStatus(StrEnum):
    """Granular verification outcome independent of raw model confidence."""

    PROVENANCE_VALIDATED = "provenance_validated"
    COMPUTATIONALLY_VERIFIED = "computationally_verified"
    INDEPENDENTLY_REVIEWED_INFERENCE = "independently_reviewed_inference"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    STALE_PENDING_REVIEW = "stale_pending_review"


class ReviewDecision(StrEnum):
    """Outcome produced by an independent reviewer."""

    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_REVISION = "needs_revision"
    UNRESOLVED = "unresolved"


class RunStatus(StrEnum):
    """Lifecycle status of an execution run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


class TaskStatus(StrEnum):
    """Lifecycle status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class EvidenceReliability(StrEnum):
    """Qualitative reliability tier of supporting evidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


class UsageMetrics(BaseModel):
    """Token, call, latency, and cost accounting for a call or run."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    actual_cost_usd: float | None = Field(default=None, ge=0.0)
    model_calls: int = Field(default=0, ge=0)
    successful_operations: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage_reported: bool = True
    returned_model: str | None = None
    provider_response_id: str | None = None
    response_status: str | None = None
    incomplete_reason: str | None = None

    @model_validator(mode="after")
    def _normalize_and_validate_tokens(self) -> UsageMetrics:
        combined = self.input_tokens + self.output_tokens
        if self.total_tokens == 0 and combined > 0:
            self.total_tokens = combined
        elif self.total_tokens < combined:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) cannot be less than "
                f"input_tokens + output_tokens ({combined})"
            )
        return self


class ClaimVersionRef(BaseModel):
    """Exact version-pinned reference to a claim in the same run."""

    claim_id: str = Field(min_length=1)
    claim_version: int = Field(ge=1)


class AgentSpec(BaseModel):
    """Specification of a dynamically assembled specialist or reviewer agent."""

    agent_id: str = Field(default_factory=lambda: _new_id("agt"))
    agent_ref: str | None = None
    run_id: str | None = None
    role: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    model_override: str | None = None
    created_by: str = Field(default="lead")
    created_at: datetime = Field(default_factory=_utc_now)


class ReviewerIdentity(BaseModel):
    """Registered independent reviewer identity within a run."""

    reviewer_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    dimensions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)


class TaskSpec(BaseModel):
    """Unit of work within an execution plan with explicit dependencies."""

    task_id: str = Field(default_factory=lambda: _new_id("tsk"))
    task_ref: str | None = None
    run_id: str | None = None
    plan_id: str | None = None
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    assigned_agent_id: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    findings: str | None = None
    error: str | None = None
    created_by: str = Field(default="lead")
    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExecutionPlan(BaseModel):
    """Structured task graph and specialist roster generated by the Lead agent."""

    plan_id: str = Field(default_factory=lambda: _new_id("pln"))
    run_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    user_requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    agents: list[AgentSpec] = Field(default_factory=list)
    tasks: list[TaskSpec] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    created_by: str = Field(default="lead")
    created_at: datetime = Field(default_factory=_utc_now)


class Evidence(BaseModel):
    """Provenance-tracked evidence supporting or refuting a claim."""

    evidence_id: str = Field(default_factory=lambda: _new_id("evd"))
    run_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    reliability: EvidenceReliability = EvidenceReliability.UNVERIFIED
    verified_in_input: bool = False
    content_hash: str | None = None
    retrieved_at: str | None = None
    source_location: str | None = None
    retrieval_status: Literal["not_applicable", "ok", "unavailable", "blocked", "failed"] = (
        "not_applicable"
    )
    task_id: str | None = None
    created_by: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utc_now)


class ToolExecutionRecord(BaseModel):
    """Immutable audit record of a deterministic tool execution."""

    tool_execution_id: str = Field(default_factory=lambda: _new_id("tlx"))
    run_id: str = Field(min_length=1)
    tool_name: str = Field(default="arithmetic", min_length=1)
    actor: str = Field(min_length=1)
    task_id: str | None = None
    expression: str = Field(min_length=1)
    variables: dict[str, str] = Field(default_factory=dict)
    input_provenance_refs: dict[str, str] = Field(default_factory=dict)
    provenance_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    proposed_result: str | None = None
    computed_result: str | None = None
    units: str = ""
    assumptions: list[str] = Field(default_factory=list)
    matches_proposed: bool | None = None
    inputs_grounded: bool = False
    has_ungrounded_literals: bool = False
    formula_appropriate: bool | None = None
    success: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class Claim(BaseModel):
    """Versioned factual, computational, or analytical claim stored in the ledger."""

    claim_id: str = Field(default_factory=lambda: _new_id("clm"))
    run_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    category: ClaimCategory = ClaimCategory.INFERENCE
    status: ClaimStatus = ClaimStatus.ASSERTED
    verification_status: VerificationStatus = VerificationStatus.UNSUPPORTED
    version: int = Field(default=1, ge=1)
    parent_claim_id: str | None = None
    author_agent_id: str = Field(min_length=1)
    task_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    depends_on_claim_ids: list[str] = Field(default_factory=list)
    premise_refs: list[ClaimVersionRef] = Field(default_factory=list)
    tool_execution_ids: list[str] = Field(default_factory=list)
    requirement_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    revision_notes: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_material: bool = True
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class AgentMessage(BaseModel):
    """Message exchanged between agents, reviewers, and Lead via MessageBus."""

    message_id: str = Field(default_factory=lambda: _new_id("msg"))
    run_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)
    recipient_id: str | None = None
    message_type: str = Field(default="statement", min_length=1)
    content: str = Field(min_length=1)
    related_claim_ids: list[str] = Field(default_factory=list)
    delivery_status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=_utc_now)


class ReviewVerdict(BaseModel):
    """Independent verification verdict targeting a specific claim version."""

    verdict_id: str = Field(default_factory=lambda: _new_id("vrd"))
    run_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    claim_version: int = Field(ge=1)
    reviewer_id: str = Field(min_length=1)
    review_dimension: str = Field(default="general", min_length=1)
    decision: ReviewDecision
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    required_correction: str | None = None
    blocking_severity: Literal["none", "minor", "blocking"] = "none"
    blocking_issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)


class RequirementCoverageItem(BaseModel):
    """Honest requirement-by-requirement linkage status."""

    requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal["answered", "partially_answered", "unresolved", "missing"]
    linked_claim_refs: list[ClaimVersionRef] = Field(default_factory=list)
    rationale: str = ""


class VerificationSummary(BaseModel):
    """Quantitative verification & honest coverage metrics surfaced on RunResult."""

    material_claim_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    unresolved_claim_count: int = Field(default=0, ge=0)
    rejected_claim_count: int = Field(default=0, ge=0)
    stale_claim_count: int = Field(default=0, ge=0)
    revision_count: int = Field(default=0, ge=0)
    evidence_types: dict[str, int] = Field(default_factory=dict)
    review_completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    requirement_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    requirements_answered: int = Field(default=0, ge=0)
    requirements_partial: int = Field(default=0, ge=0)
    requirements_unresolved: int = Field(default=0, ge=0)
    requirements_missing: int = Field(default=0, ge=0)
    requirement_statuses: list[RequirementCoverageItem] = Field(default_factory=list)
    execution_completed: bool = False
    release_checks_executed: bool = True
    content_released_safely: bool = False
    coverage_methodology: str = "internal_claim_linkage_not_independent_grade"
    independent_grade: float | None = Field(default=None, ge=0.0, le=1.0)
    honesty_gate_passed: bool = False
    honesty_gate_reasons: list[str] = Field(default_factory=list)


class HonestyGateRecord(BaseModel):
    """Persisted deterministic Honesty Gate decision for a run."""

    gate_id: str = Field(default_factory=lambda: _new_id("hgt"))
    run_id: str = Field(min_length=1)
    passed: bool
    release_mode: Literal["full", "partial", "blocked"]
    rejected_segments: list[dict[str, Any]] = Field(default_factory=list)
    rejected_claim_refs: list[dict[str, Any]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    verification_summary: VerificationSummary = Field(default_factory=VerificationSummary)
    created_at: datetime = Field(default_factory=_utc_now)


class RunConfig(BaseModel):
    """Immutable configuration snapshot captured for an individual run."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0.0)
    run_deadline_seconds: float = Field(default=300.0, gt=0.0)
    max_retries: int = Field(ge=0)
    max_concurrency: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_model_calls_per_run: int = Field(ge=1)
    max_cost_per_run_usd: float = Field(ge=0.0)
    max_cumulative_cost_usd: float = Field(ge=0.0)
    reasoning_effort: str | None = None


class ExecutionEvent(BaseModel):
    """Audit trail event persisted for observability and trace retrieval."""

    event_id: str = Field(default_factory=lambda: _new_id("evt"))
    run_id: str = Field(min_length=1)
    seq: int | None = Field(default=None, ge=1)
    event_type: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    message: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class RunResult(BaseModel):
    """Structured final result returned by a baseline or network run."""

    run_id: str = Field(default_factory=lambda: _new_id("run"))
    mode: Literal["baseline", "network"] = "baseline"
    status: RunStatus
    prompt: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    returned_model: str | None = None
    answer: str = ""
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence_notes: str | None = None
    referenced_claim_ids: list[str] = Field(default_factory=list)
    specialist_roles: list[str] = Field(default_factory=list)
    incomplete_tasks: list[str] = Field(default_factory=list)
    synthesis_mode: str | None = None
    baseline_mode: str | None = None
    verification: VerificationSummary | None = None
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None


class RunTrace(BaseModel):
    """Complete persisted trace returned by the trace tool."""

    run_id: str
    mode: Literal["baseline", "network"]
    status: RunStatus
    prompt: str
    provider: str
    model: str
    config: RunConfig
    result: RunResult | None = None
    plan: ExecutionPlan | None = None
    agents: list[AgentSpec] = Field(default_factory=list)
    reviewers: list[ReviewerIdentity] = Field(default_factory=list)
    tasks: list[TaskSpec] = Field(default_factory=list)
    task_transitions: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    review_verdicts: list[ReviewVerdict] = Field(default_factory=list)
    honesty_gate: HonestyGateRecord | None = None
    events: list[ExecutionEvent] = Field(default_factory=list)
    usage_records: list[dict[str, Any]] = Field(default_factory=list)
    budget_reservations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Compact model-facing schemas (separated from persistence IDs & timestamps)
# ---------------------------------------------------------------------------


class PlannedAgentSpec(BaseModel):
    """Compact model-facing specialist definition produced by Lead planner."""

    agent_ref: str = Field(min_length=1, max_length=24)
    role: str = Field(min_length=1, max_length=120)
    instructions: str = Field(min_length=1, max_length=1200)
    capabilities: list[str] = Field(default_factory=list)
    model_override: str | None = None


class PlannedTaskSpec(BaseModel):
    """Compact model-facing task node produced by Lead planner."""

    task_ref: str = Field(min_length=1, max_length=24)
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=1500)
    assigned_agent_ref: str = Field(min_length=1, max_length=24)
    depends_on: list[str] = Field(default_factory=list)
    expected_deliverables: list[str] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    """Structured plan returned directly by the Lead planning model call."""

    goal: str = Field(min_length=1)
    user_requirements: list[str] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    agents: list[PlannedAgentSpec] = Field(min_length=1)
    tasks: list[PlannedTaskSpec] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class ArithmeticVariableBinding(BaseModel):
    """Portable schema binding pairing a variable name, value, and provenance alias."""

    name: str = Field(min_length=1)
    value: str | int | float
    provenance_ref: str = Field(default="")


class ArithmeticRequest(BaseModel):
    """Structured deterministic computation request emitted by baseline or specialists."""

    expression: str = Field(min_length=1, max_length=256)
    variable_bindings: list[ArithmeticVariableBinding] = Field(default_factory=list)
    proposed_result: str | int | float | None = None
    units: str = ""
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_variable_dicts(cls, data: Any) -> Any:
        if isinstance(data, dict) and "variable_bindings" not in data and "variables" in data:
            raw_vars = data.get("variables") or {}
            raw_refs = data.get("input_provenance_refs") or {}
            if isinstance(raw_vars, dict):
                bindings = [
                    {
                        "name": k,
                        "value": v,
                        "provenance_ref": str(raw_refs.get(k, ""))
                        if isinstance(raw_refs, dict)
                        else "",
                    }
                    for k, v in raw_vars.items()
                ]
                copied = {
                    k: v for k, v in data.items() if k not in {"variables", "input_provenance_refs"}
                }
                copied["variable_bindings"] = bindings
                return copied
        return data

    @property
    def variables(self) -> dict[str, str | int | float]:
        """Dictionary mapping variable name -> value for tool evaluation."""
        return {b.name: b.value for b in self.variable_bindings}

    @property
    def input_provenance_refs(self) -> dict[str, str]:
        """Dictionary mapping variable name -> provenance reference target."""
        return {b.name: b.provenance_ref for b in self.variable_bindings}


class ProposedEvidence(BaseModel):
    """Compact evidence proposal emitted by a specialist."""

    source_type: Literal[
        "user_prompt",
        "dependency_finding",
        "calculation",
        "external_retrieved",
        "external_unverified",
    ]
    source_ref: str = Field(min_length=1, max_length=120)
    excerpt: str = Field(min_length=1, max_length=360)


class ProposedClaim(BaseModel):
    """Compact atomic claim assertion emitted by a specialist."""

    claim_ref: str | None = Field(default=None, max_length=32)
    statement: str = Field(min_length=1, max_length=360)
    category: ClaimCategory = ClaimCategory.INFERENCE
    evidence_indices: list[int] = Field(default_factory=list)
    premise_claim_ids: list[str] = Field(default_factory=list)
    computation: ArithmeticRequest | None = None
    requirement_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_unresolved: bool = False


class OutgoingMessageInstruction(BaseModel):
    """Structured peer message emitted by a specialist."""

    recipient_ref: str | None = Field(
        default=None,
        description="Target agent_ref, 'lead', or null for broadcast.",
    )
    message_type: Literal["finding", "question", "clarification", "warning"] = "finding"
    content: str = Field(min_length=1, max_length=500)
    related_claim_indices: list[int] = Field(default_factory=list)


class SpecialistOutput(BaseModel):
    """Compact structured result returned by a specialist task execution."""

    findings: str = Field(
        min_length=1,
        max_length=1500,
        description="Concise task findings (no prompt repetition).",
    )
    claims: list[ProposedClaim] = Field(
        default_factory=list,
        description=(
            "Atomic claims covering all assigned deliverables and independent numerical results "
            "(one claim per distinct result, scaled to task complexity)."
        ),
    )
    evidence_proposals: list[ProposedEvidence] = Field(
        default_factory=list,
        description="Short verbatim evidence excerpts supporting claims.",
    )
    assumptions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    outgoing_messages: list[OutgoingMessageInstruction] = Field(default_factory=list)
    task_completed: bool = True


class TargetedClaimRevision(BaseModel):
    """Explicit claim-ID and version-targeted revision emitted by a specialist."""

    target_claim_id: str = Field(min_length=1)
    expected_version: int = Field(ge=1)
    new_statement: str = Field(min_length=1, max_length=360)
    category: ClaimCategory | None = None
    evidence_indices: list[int] = Field(default_factory=list)
    premise_claim_ids: list[str] = Field(default_factory=list)
    computation: ArithmeticRequest | None = None
    assumptions: list[str] = Field(default_factory=list)
    revision_notes: str = Field(min_length=1, max_length=300)
    is_unresolved: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class FollowUpOutput(BaseModel):
    """Compact structured output from a bounded peer-communication checkpoint."""

    updated_findings: str = Field(min_length=1, max_length=1500)
    revised_claims: list[TargetedClaimRevision] = Field(default_factory=list)
    new_claims: list[ProposedClaim] = Field(default_factory=list)
    evidence_proposals: list[ProposedEvidence] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    outgoing_messages: list[OutgoingMessageInstruction] = Field(default_factory=list)


class RevisionOutput(BaseModel):
    """Compact structured output from targeted dispute-resolution repair."""

    updated_findings: str = Field(min_length=1, max_length=1500)
    revisions: list[TargetedClaimRevision] = Field(default_factory=list)
    evidence_proposals: list[ProposedEvidence] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    outgoing_messages: list[OutgoingMessageInstruction] = Field(default_factory=list)


class ReviewerVerdictItem(BaseModel):
    """Model-facing verdict entry for a specific claim alias or claim ID and version."""

    claim_id: str = Field(min_length=1)
    claim_version: int | None = Field(default=None, ge=1)
    review_dimension: str = Field(default="combined_evidence_consistency", min_length=1)
    decision: ReviewDecision
    rationale: str = Field(min_length=1, max_length=250)
    evidence_refs: list[str] = Field(default_factory=list)
    required_correction: str | None = Field(default=None, max_length=200)
    blocking_severity: Literal["none", "minor", "blocking"] = "none"


class ReviewerBatchOutput(BaseModel):
    """Batched independent reviewer output."""

    verdicts: list[ReviewerVerdictItem] = Field(default_factory=list)
    missing_requirement_coverage: list[str] = Field(default_factory=list)


class ReleaseSegment(BaseModel):
    """Structured release segment audited by the deterministic Honesty Gate."""

    segment_type: Literal[
        "requirement_answer",
        "accepted_claim",
        "conditional_conclusion",
        "recommendation",
        "assumption",
        "unresolved_issue",
        "missing_information",
    ]
    text: str = Field(default="", max_length=1000)
    claim_refs: list[ClaimVersionRef] = Field(default_factory=list)
    requirement_ref: str | None = None
    is_factual_assertion: bool = True


class SynthesisOutput(BaseModel):
    """Evidence-constrained structured release output produced by Lead synthesizer."""

    user_requirements: list[str] = Field(default_factory=list)
    accepted_segments: list[ReleaseSegment] = Field(default_factory=list)
    conditional_conclusions: list[ReleaseSegment] = Field(default_factory=list)
    recommendations: list[ReleaseSegment] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    new_factual_assertions: list[str] = Field(default_factory=list)
    referenced_claim_ids: list[str] = Field(default_factory=list)
    confidence_notes: str | None = None
