"""Evaluation contracts, schemas, manifests, and rubric definitions for CrewForge."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from crewforge.schemas import RunResult


class PromptDomain(StrEnum):
    """Evaluation domain categories."""

    FINANCIAL = "financial"
    INCIDENT_RESPONSE = "incident_response"
    CAUSATION = "causation"
    DOCUMENT_GROUNDED = "document_grounded"
    STRATEGIC_PLANNING = "strategic_planning"


class PromptDifficulty(StrEnum):
    """Prompt complexity classification."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AblationVariant(StrEnum):
    """Evaluation-only ablation variants."""

    FULL = "full"
    NO_REVIEWER = "no_reviewer"
    NO_REPAIR = "no_repair"
    NO_PEER_MESSAGES = "no_peer_messages"


class BenchmarkItem(BaseModel):
    """One immutable benchmark evaluation case."""

    prompt_id: str = Field(min_length=1)
    title: str = ""
    domain: PromptDomain
    difficulty: PromptDifficulty = PromptDifficulty.MEDIUM
    prompt: str = Field(min_length=1)
    prompt_hash: str = ""
    reference_documents: list[str] = Field(default_factory=list)
    document_hashes: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.prompt_hash and self.prompt:
            self.prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


class BenchmarkManifest(BaseModel):
    """Collection of evaluation prompts forming a frozen dataset."""

    manifest_id: str
    name: str
    description: str = ""
    version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit_sha: str = ""
    items: list[BenchmarkItem] = Field(default_factory=list)

    def get_item(self, prompt_id: str) -> BenchmarkItem | None:
        for it in self.items:
            if it.prompt_id == prompt_id:
                return it
        return None


class ExpectedNumericalCheck(BaseModel):
    """Deterministic objective check for an exact numeric quantity."""

    variable_name: str
    expected_value: float
    tolerance: float = 1e-4
    is_percentage: bool = False
    is_percentage_points: bool = False
    expected_units: str = ""
    points: float = 10.0
    requirement_ref: str | None = None
    description: str = ""


class ExpectedLogicalCheck(BaseModel):
    """Deterministic check for required conclusions and forbidden contradictions."""

    check_id: str
    description: str
    required_phrases: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    required_assumptions_or_conditions: list[str] = Field(default_factory=list)
    points: float = 5.0
    requirement_ref: str | None = None


class ExpectedMissingInfoCheck(BaseModel):
    """Deterministic check that missing context was accurately identified."""

    check_id: str
    description: str
    required_missing_concepts: list[str] = Field(default_factory=list)
    points: float = 5.0
    requirement_ref: str | None = None


class EvaluationRubric(BaseModel):
    """100-point independent evaluation rubric for a specific prompt."""

    prompt_id: str
    domain: PromptDomain

    # 100-point rubric breakdown
    factual_correctness_max: float = 30.0
    requirement_coverage_max: float = 25.0
    evidence_grounding_max: float = 20.0
    internal_consistency_max: float = 15.0
    usefulness_and_uncertainty_max: float = 10.0
    max_unsupported_assertion_penalty: float = 20.0

    numerical_checks: list[ExpectedNumericalCheck] = Field(default_factory=list)
    logical_checks: list[ExpectedLogicalCheck] = Field(default_factory=list)
    missing_info_checks: list[ExpectedMissingInfoCheck] = Field(default_factory=list)
    scoring_notes: str = ""


class ObjectiveCheckResult(BaseModel):
    """Result of one deterministic assertion."""

    check_name: str
    passed: bool
    score: float
    max_score: float
    details: str = ""


class EvaluationScore(BaseModel):
    """Full 100-point evaluation score for one answer."""

    prompt_id: str
    system_label: str
    factual_correctness: float = Field(ge=0.0, le=30.0)
    requirement_coverage: float = Field(ge=0.0, le=25.0)
    evidence_grounding: float = Field(ge=0.0, le=20.0)
    internal_consistency: float = Field(ge=0.0, le=15.0)
    usefulness_and_uncertainty: float = Field(ge=0.0, le=10.0)
    unsupported_assertion_penalty: float = Field(ge=0.0, le=20.0)
    total_score: float = Field(ge=0.0, le=100.0)

    objective_score: float = 0.0
    objective_max_score: float = 0.0
    objective_checks: list[ObjectiveCheckResult] = Field(default_factory=list)
    evaluator_notes: str = ""
    is_fully_objective: bool = False


class BlindAnswer(BaseModel):
    """Clean, presentation-neutral answer for blind evaluation."""

    label: Literal["A", "B"]
    content: str
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence_notes: str | None = None


class BlindEvaluationItem(BaseModel):
    """A/B blinded item for human or objective evaluation."""

    prompt_id: str
    domain: PromptDomain
    prompt: str
    reference_documents: list[str] = Field(default_factory=list)
    answer_a: BlindAnswer
    answer_b: BlindAnswer


class BlindKey(BaseModel):
    """Separate unblinding key held until scores are finalized."""

    prompt_id: str
    label_a_system: str
    label_b_system: str
    run_id_a: str
    run_id_b: str
    salt: str


class EvaluatorPacket(BaseModel):
    """Complete blind evaluation packet exported for human or independent evaluation."""

    packet_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    items: list[BlindEvaluationItem] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """One completed benchmark comparison between baseline and network."""

    prompt_id: str
    domain: PromptDomain
    baseline_run_id: str
    network_run_id: str
    baseline_result: RunResult
    network_result: RunResult
    baseline_score: EvaluationScore | None = None
    network_score: EvaluationScore | None = None
    latency_difference_ms: float = 0.0
    cost_difference_usd: float = 0.0
    git_commit_sha: str = ""
