"""Subprocess entrypoint launching CrewForge MCP server with FakeProvider for offline tests."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.mcp_server import create_mcp_server
from crewforge.schemas import (
    FollowUpOutput,
    PlannerOutput,
    ReviewerBatchOutput,
    SpecialistOutput,
    SynthesisOutput,
)
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider

_CLAIM_HEADER_RE = re.compile(r"Claim (clm_[a-f0-9]+) \(version=(\d+),")
_SUPPORTED_CLAIM_RE = re.compile(r"- (clm_[a-f0-9]+) \(version=(\d+),.*?status=supported")


def _scripted_mcp_handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
    if response_model is PlannerOutput:
        return {
            "goal": "Analyze prompt via dynamic specialists.",
            "user_requirements": ["Address schema validation in multi-agent systems"],
            "acceptance_criteria": ["Provide verified findings and surface missing data."],
            "agents": [
                {
                    "agent_ref": "a1",
                    "role": "Primary Analyst",
                    "instructions": "Analyze core problem aspects.",
                    "capabilities": ["analysis", "reasoning"],
                },
                {
                    "agent_ref": "a2",
                    "role": "Risk Reviewer",
                    "instructions": "Identify edge cases and missing information.",
                    "capabilities": ["critique"],
                },
            ],
            "tasks": [
                {
                    "task_ref": "t1",
                    "title": "Core analysis",
                    "description": "Analyze the user request.",
                    "assigned_agent_ref": "a1",
                    "depends_on": [],
                    "expected_deliverables": ["Core analytical findings"],
                },
                {
                    "task_ref": "t2",
                    "title": "Risk and gap check",
                    "description": "Review core analysis and highlight missing info.",
                    "assigned_agent_ref": "a2",
                    "depends_on": ["t1"],
                    "expected_deliverables": ["Gap assessment"],
                },
            ],
            "assumptions": ["Evaluated offline in test mode."],
            "missing_information": ["Live operational context not provided."],
        }
    if response_model is SpecialistOutput:
        if "Assigned task (t1)" in prompt:
            return {
                "findings": "Core finding: schema validation prevents silent data corruption.",
                "claims": [
                    {
                        "statement": "Schema validation is required in multi-agent systems.",
                        "category": "given_fact",
                        "evidence_indices": [0],
                        "assumptions": [],
                        "confidence": 0.9,
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "user_prompt",
                        "source_ref": "prompt",
                        "excerpt": "schema validation",
                    }
                ],
                "assumptions": ["Inputs follow JSON contracts."],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [
                    {
                        "recipient_ref": "a2",
                        "message_type": "finding",
                        "content": "Check edge-case contract drift for t1.",
                        "related_claim_indices": [0],
                    }
                ],
                "task_completed": True,
            }
        return {
            "findings": "Risk review confirmed contract drift risk and acknowledged message.",
            "claims": [
                {
                    "statement": "Unvalidated responses can propagate invalid state in systems.",
                    "category": "logical_inference",
                    "evidence_indices": [],
                    "assumptions": ["No external validator exists downstream"],
                    "confidence": 0.85,
                }
            ],
            "evidence_proposals": [],
            "assumptions": [],
            "uncertainties": ["Runtime load profile unknown."],
            "missing_information": ["Target throughput constraints."],
            "outgoing_messages": [
                {
                    "recipient_ref": "lead",
                    "message_type": "finding",
                    "content": "Risk review complete for schema validation.",
                    "related_claim_indices": [0],
                }
            ],
            "task_completed": True,
        }
    if response_model is FollowUpOutput:
        return {
            "updated_findings": "Follow-up completed.",
            "revised_claims": [],
            "new_claims": [],
            "evidence_proposals": [],
            "uncertainties": [],
            "outgoing_messages": [],
        }
    if response_model is ReviewerBatchOutput:
        verdicts = []
        for cid, cver in _CLAIM_HEADER_RE.findall(prompt):
            verdicts.append(
                {
                    "claim_id": cid,
                    "claim_version": int(cver),
                    "review_dimension": "combined_evidence_consistency",
                    "decision": "approve",
                    "rationale": "Consistent with prompt and verified evidence.",
                    "evidence_refs": [],
                    "blocking_severity": "none",
                }
            )
        return {"verdicts": verdicts, "missing_requirement_coverage": []}
    if response_model is SynthesisOutput:
        supported_refs = re.findall(
            r"- (clm_[a-f0-9]+) \(version=(\d+),.*?status=supported.*?\): ([^\n]+)",
            prompt,
        )
        accepted = []
        for cid, cver, stmt in supported_refs:
            accepted.append(
                {
                    "segment_type": "accepted_claim",
                    "text": stmt.strip(),
                    "claim_refs": [{"claim_id": cid, "claim_version": int(cver)}],
                    "is_factual_assertion": True,
                }
            )
        return {
            "user_requirements": ["Address schema validation in multi-agent systems"],
            "accepted_segments": accepted,
            "conditional_conclusions": [],
            "recommendations": [],
            "assumptions": ["Evaluated offline."],
            "unresolved_issues": [],
            "missing_information": ["Target throughput constraints."],
            "new_factual_assertions": [],
            "referenced_claim_ids": [cid for cid, _, _ in supported_refs],
            "confidence_notes": "Synthesized from reviewed claims.",
        }
    return {
        "answer": f"Deterministic baseline answer for: {prompt}",
        "missing_information": ["No live runtime environment metrics were provided."],
        "assumptions": ["Evaluated offline using deterministic test provider."],
        "confidence_notes": "Deterministic test output.",
    }


def main() -> None:
    db_path = Path(os.environ["CREWFORGE_DB_PATH"])
    settings = CrewForgeSettings(_env_file=None, db_path=db_path, max_retries=0)
    database = Database(db_path)
    provider = FakeProvider(
        provider_name="fake-mcp",
        model_name="fake-mcp-model",
        is_paid_provider=False,
        max_retries=0,
        handler=_scripted_mcp_handler,
    )
    server = create_mcp_server(
        settings=settings,
        database=database,
        provider=provider,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
