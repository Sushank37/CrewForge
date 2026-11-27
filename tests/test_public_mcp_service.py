"""Offline tests for the dedicated unauthenticated public evaluation MCP service."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.public_mcp_service import PublicJobManager, create_public_mcp_app
from tests.fake_provider import FakeProvider


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _scripted_handler(prompt: str, response_model: Any) -> Any:
    schema_name = response_model.__name__ if response_model is not None else ""
    if schema_name in {"ExecutionPlan", "PlannerOutput"}:
        return {
            "goal": "Analyze reference doc",
            "user_requirements": ["Verify the reference metric"],
            "acceptance_criteria": ["Grounded answer"],
            "agents": [
                {
                    "agent_ref": "a1",
                    "role": "Research Analyst",
                    "specialty": "Verification",
                    "instructions": "Verify facts from doc_1.",
                    "capabilities": ["reasoning"],
                }
            ],
            "tasks": [
                {
                    "task_ref": "t1",
                    "title": "Verify doc_1",
                    "description": "Extract metric from doc_1.",
                    "assigned_agent_ref": "a1",
                    "depends_on": [],
                    "expected_deliverables": ["Metric verified"],
                }
            ],
            "assumptions": [],
            "missing_information": [],
        }
    if schema_name in {"WorkerOutput", "SpecialistOutput"}:
        return {
            "findings": "Verified budget ceiling is 420 USD from doc_1.",
            "evidence": [
                {
                    "source_type": "external_retrieved",
                    "source_ref": "doc_1",
                    "excerpt": "The budget ceiling is 420 USD.",
                }
            ],
            "claims": [
                {
                    "statement": "The documented budget ceiling is 420 USD.",
                    "category": "external_factual_assertion",
                    "evidence_indices": [0],
                    "requirement_refs": ["req_1"],
                }
            ],
            "assumptions": [],
            "uncertainties": [],
            "missing_information": [],
            "outgoing_messages": [],
        }
    if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
        cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
        eids = list(dict.fromkeys(re.findall(r"(evd_[A-Za-z0-9_]+)", prompt)))
        return {
            "verdicts": [
                {
                    "claim_id": cid,
                    "decision": "approve",
                    "rationale": "Supported by doc_1.",
                    "evidence_refs": eids[:1],
                    "required_correction": None,
                    "blocking_severity": "none",
                    "blocking_issues": [],
                }
                for cid in cids
            ]
        }
    if schema_name == "SynthesisOutput":
        m = re.search(r"(clm_[A-Za-z0-9_]+)", prompt)
        cid = m.group(1) if m else "clm_missing"
        ver = 1
        return {
            "direct_summary": "The documented budget ceiling is 420 USD.",
            "accepted_segments": [
                {
                    "segment_type": "accepted_claim",
                    "text": "The documented budget ceiling is 420 USD.",
                    "claim_refs": [{"claim_id": cid, "claim_version": ver}],
                    "requirement_ref": "req_1",
                }
            ],
            "conditional_segments": [],
            "recommendations": [],
            "new_factual_assertions": [],
            "referenced_claim_ids": [cid],
            "assumptions": [],
            "missing_information": [],
        }
    if schema_name == "BaselineModelOutput":
        return {
            "answer": "Baseline: The documented budget ceiling is 420 USD.",
            "missing_information": [],
            "assumptions": [],
            "computations": [],
            "confidence_notes": "Baseline draft.",
        }
    if schema_name == "BaselineCritiqueOutput":
        return {
            "unanswered_requirements": [],
            "arithmetic_issues": [],
            "missing_context_or_assumptions": [],
            "is_fully_satisfied": True,
            "computations": [],
            "refined_answer": "Baseline: The documented budget ceiling is 420 USD.",
            "missing_information": [],
            "assumptions": [],
            "confidence_notes": "Verified.",
        }
    raise AssertionError(f"Unexpected schema: {schema_name}")


async def test_public_mcp_unauthenticated_lifecycle_baseline_cancel_and_isolation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "public_mcp.db"
    budget_path = tmp_path / "public_budget.db"
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-public-model",
        openai_api_key="sk-secret-team-key-should-never-leak",
        db_path=db_path,
        budget_authority_path=budget_path,
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
    )
    app = create_public_mcp_app(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_path,
        provider_factory=lambda s: FakeProvider(
            handler=_scripted_handler,
            provider_name="fake-public",
            model_name=s.model,
            delay_seconds=0.15,
        ),
        max_concurrent_jobs=2,
    )

    # 1. Public /health works unauthenticated; private admin/chat routes do NOT exist
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://public.crewforge.test"
    ) as http_probe:
        health = await http_probe.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert "sk-secret-team-key" not in health.text
        assert (await http_probe.get("/api/admin/tokens")).status_code == 404
        assert (await http_probe.get("/api/chats")).status_code == 404
        assert (await http_probe.get("/api/settings")).status_code == 404

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve_task = asyncio.create_task(server.serve())
    try:
        for _ in range(50):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started

        base_url = f"http://127.0.0.1:{port}"
        # 2. Standard unauthenticated MCP client connects with ONLY the endpoint URL
        async with (
            httpx.AsyncClient(timeout=30.0) as http_client,
            streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_names = {t.name for t in tools_resp.tools}
            assert tool_names == {
                "run_task",
                "get_task_status",
                "get_task_result",
                "get_task_trace",
                "run_single_agent",
                "cancel_task",
            }

            # 3. Enqueue network task -> returns unguessable task_id immediately (< 1s)
            t0 = time.perf_counter()
            run_resp = await session.call_tool(
                "run_task",
                {
                    "prompt": "Verify the documented budget ceiling.",
                    "reference_documents": [
                        {"title": "Budget Policy", "content": "The budget ceiling is 420 USD."}
                    ],
                },
            )
            elapsed = time.perf_counter() - t0
            assert not run_resp.isError
            assert elapsed < 2.0
            run_data = json.loads(run_resp.content[0].text)
            task_id = run_data["task_id"]
            assert task_id.startswith("tsk_pub_")
            assert len(task_id) == len("tsk_pub_") + 32
            assert run_data["status"] == "queued"

            # Poll status until completed
            final_status = None
            for _ in range(40):
                st_resp = await session.call_tool("get_task_status", {"task_id": task_id})
                st_data = json.loads(st_resp.content[0].text)
                assert st_data["found"] is True
                final_status = st_data["status"]
                if final_status in {"completed", "partial", "failed"}:
                    break
                await asyncio.sleep(0.1)
            assert final_status in {"completed", "partial"}, st_data.get("error")

            # Retrieve result + sources + verification
            res_resp = await session.call_tool("get_task_result", {"task_id": task_id})
            res_data = json.loads(res_resp.content[0].text)
            assert res_data["ready"] is True
            assert res_data["verification"]["independent_verification_performed"] is True
            assert res_data["verification"]["material_claim_count"] >= 1
            assert len(res_data["sources"]) >= 1
            assert "sk-secret-team-key" not in json.dumps(res_data)

            # Retrieve sanitized trace
            tr_resp = await session.call_tool("get_task_trace", {"task_id": task_id})
            assert not tr_resp.isError, tr_resp.content[0].text
            tr_data = json.loads(tr_resp.content[0].text)
            assert tr_data["found"] is True
            assert len(tr_data["agents"]) == 1
            assert len(tr_data["assignments"]) == 1
            assert len(tr_data["review_decisions"]) >= 1
            assert len(tr_data["claims_and_evidence_lineage"]) >= 1
            assert "sk-secret-team-key" not in json.dumps(tr_data)

            # 4. Single-agent baseline comparison
            base_resp = await session.call_tool(
                "run_single_agent",
                {
                    "prompt": "Verify the documented budget ceiling.",
                    "reference_documents": ["The budget ceiling is 420 USD."],
                },
            )
            base_id = json.loads(base_resp.content[0].text)["task_id"]
            for _ in range(40):
                st = json.loads(
                    (await session.call_tool("get_task_status", {"task_id": base_id}))
                    .content[0]
                    .text
                )
                if st["status"] in {"completed", "partial", "failed"}:
                    break
                await asyncio.sleep(0.1)
            base_res = json.loads(
                (await session.call_tool("get_task_result", {"task_id": base_id}))
                .content[0]
                .text
            )
            assert base_res["mode"] == "baseline"
            assert "420 USD" in base_res["answer"]
            assert base_res["verification"]["independent_verification_performed"] is False

            # 5. Cancellation of queued/running job
            c_run = await session.call_tool("run_task", {"prompt": "Long running task to cancel"})
            c_id = json.loads(c_run.content[0].text)["task_id"]
            c_res = json.loads(
                (await session.call_tool("cancel_task", {"task_id": c_id})).content[0].text
            )
            assert c_res["cancelled"] is True
            assert c_res["status"] == "cancelled"

            # 6. Unknown task_id handled cleanly
            unk = json.loads(
                (await session.call_tool("get_task_status", {"task_id": "tsk_pub_unknown"}))
                .content[0]
                .text
            )
            assert unk["found"] is False
    finally:
        server.should_exit = True
        await serve_task


def test_public_job_manager_restart_recovery_prevents_duplicate_execution(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "restart.db"
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-model",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
    )
    PublicJobManager(settings=settings, db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO public_jobs (
                task_id, mode, status, prompt, reference_documents_json,
                current_stage, progress, provider, model,
                result_json, error, created_at, updated_at, completed_at
            ) VALUES (
                'tsk_pub_inflight123', 'network', 'running', 'Inflight prompt', '[]',
                'specialist_execution', 0.4, 'openai', 'fake-model',
                NULL, NULL, '2026-11-27T00:00:00Z', '2026-11-27T00:00:00Z', NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    # Restarting PublicJobManager marks interrupted jobs failed without re-executing them
    mgr2 = PublicJobManager(settings=settings, db_path=db_path)
    row = mgr2._get_job_row("tsk_pub_inflight123")
    assert row is not None
    assert row["status"] == "failed"
    assert row["current_stage"] == "interrupted_by_restart"
    assert "duplicate execution was prevented" in row["error"]
