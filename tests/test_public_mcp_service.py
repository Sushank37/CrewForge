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

from crewforge.config import CrewForgeSettings, ModelPricing, ProviderType
from crewforge.public_mcp_service import PublicJobManager, create_public_mcp_app
from tests.fake_provider import ChecklistFixtureProvider as FakeProvider


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
    if schema_name in {"WorkerOutput", "SpecialistOutput", "ReconciliationOutput"}:
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


async def test_public_mcp_hardening_bounds_idempotency_safe_trace_and_external_host(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "hardened.db"
    budget_path = tmp_path / "hardened_budget.db"
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-public-model",
        openai_api_key="sk-secret",
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
            delay_seconds=0.25,
        ),
        max_concurrent_jobs=1,
        max_pending_jobs=1,
    )

    # 1. Request size > 512 KB returns 413
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://crewforge-eval.up.railway.app",
    ) as client:
        oversized = await client.post(
            "/mcp",
            headers={"Host": "crewforge-eval.up.railway.app"},
            content=b"x" * (513 * 1024),
        )
        assert oversized.status_code == 413

    # 2. External Host header works cleanly over Streamable HTTP without auth
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
        ext_headers = {
            "Host": "crewforge-eval.up.railway.app",
            "X-Forwarded-Proto": "https",
        }
        async with (
            httpx.AsyncClient(timeout=30.0, headers=ext_headers) as http_client,
            streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            # Submit with idempotency_key
            first = json.loads(
                (
                    await session.call_tool(
                        "run_task",
                        {
                            "prompt": "Verify the documented budget ceiling.",
                            "idempotency_key": "judge-idem-001",
                        },
                    )
                )
                .content[0]
                .text
            )
            assert first["accepted"] is True
            tid = first["task_id"]

            # Safe trace for queued/running job before completion
            queued_trace = json.loads(
                (await session.call_tool("get_task_trace", {"task_id": tid}))
                .content[0]
                .text
            )
            assert queued_trace["found"] is True

            # Duplicate idempotency_key returns same task_id without re-enqueuing
            dup = json.loads(
                (
                    await session.call_tool(
                        "run_task",
                        {
                            "prompt": "Verify the documented budget ceiling.",
                            "idempotency_key": "judge-idem-001",
                        },
                    )
                )
                .content[0]
                .text
            )
            assert dup["deduplicated"] is True
            assert dup["task_id"] == tid

            # Conflicting inputs with same idempotency_key rejected
            conflict = json.loads(
                (
                    await session.call_tool(
                        "run_task",
                        {
                            "prompt": "Different prompt",
                            "idempotency_key": "judge-idem-001",
                        },
                    )
                )
                .content[0]
                .text
            )
            assert conflict["accepted"] is False
            assert conflict["error_code"] == "idempotency_conflict"

            # Capacity exhausted (max_pending_jobs=1) returns clear retryable error
            cap = json.loads(
                (
                    await session.call_tool(
                        "run_task",
                        {"prompt": "Second concurrent task exceeds capacity"},
                    )
                )
                .content[0]
                .text
            )
            assert cap["accepted"] is False
            assert cap["retryable"] is True
            assert cap["error_code"] == "capacity_exhausted"

            # Wait for first task to finish, then verify cancelled trace is safe
            for _ in range(40):
                st = json.loads(
                    (await session.call_tool("get_task_status", {"task_id": tid}))
                    .content[0]
                    .text
                )
                if st["status"] in {"completed", "partial", "failed"}:
                    break
                await asyncio.sleep(0.1)
    finally:
        server.should_exit = True
        await serve_task


async def test_public_mcp_shared_spending_safeguards_under_concurrent_submissions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "conc_budget.db"
    budget_path = tmp_path / "conc_authority.db"
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-paid-model",
        openai_api_key="sk-offline-test",
        db_path=db_path,
        budget_authority_path=budget_path,
        allow_paid_runs=True,
        max_cost_per_run_usd=0.05,
        max_cumulative_cost_usd=0.06,
        total_budget_ceiling_usd=0.06,
        protected_reserve_usd=0.0,
        pricing_json={
            "fake-paid-model": ModelPricing(input_per_1m=1000.0, output_per_1m=1000.0)
        },
    )
    mgr = PublicJobManager(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_path,
        provider_factory=lambda s: FakeProvider(
            handler=_scripted_handler,
            provider_name="fake-paid",
            model_name=s.model,
            is_paid_provider=True,
            delay_seconds=0.05,
        ),
        max_concurrent_jobs=4,
        max_pending_jobs=8,
    )
    submissions = await asyncio.gather(
        *[
            mgr.enqueue_task(
                prompt=f"Concurrent task {i}",
                mode="network",
            )
            for i in range(4)
        ]
    )
    task_ids = [s["task_id"] for s in submissions if s.get("accepted")]
    assert len(task_ids) == 4
    for _ in range(60):
        statuses = [
            (await mgr.get_task_status(tid))["status"] for tid in task_ids
        ]
        if all(st in {"completed", "partial", "failed", "cancelled"} for st in statuses):
            break
        await asyncio.sleep(0.05)

    results = [await mgr.get_task_result(tid) for tid in task_ids]
    total_spent = sum(
        float((r.get("usage") or {}).get("estimated_cost_usd", 0.0)) for r in results
    )
    assert total_spent <= 0.06 + 1e-9
    # Verify failed/cancelled jobs without review report independent_verification_performed=False
    for r in results:
        if r["status"] == "failed" and r.get("verification"):
            assert r["verification"]["independent_verification_performed"] is False
    await mgr.shutdown()


async def test_public_mcp_production_openai_provider_network_and_baseline_with_default_pricing(
    tmp_path: Path,
) -> None:
    """Reproduce Railway cloud profile + adaptive capacity + DEFAULT_MODEL_PRICING."""
    from types import SimpleNamespace

    from crewforge.providers.openai_provider import OpenAIProvider
    from tests.fake_provider import checklist_fixture

    fixture_handler = checklist_fixture(_scripted_handler)

    class _FakeResponses:
        async def create(self, **kwargs: Any) -> Any:
            schema_name = (
                kwargs.get("text", {}).get("format", {}).get("name", "")
                if isinstance(kwargs.get("text"), dict)
                else ""
            )
            prompt = str(kwargs.get("input", ""))
            if schema_name in {"ExecutionPlan", "PlannerOutput"}:
                payload = fixture_handler(prompt, type("PlannerOutput", (), {}))
            elif schema_name in {"WorkerOutput", "SpecialistOutput"}:
                payload = fixture_handler(prompt, type("SpecialistOutput", (), {}))
            elif schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
                payload = fixture_handler(prompt, type("ReviewerBatchOutput", (), {}))
            elif schema_name == "SynthesisOutput":
                payload = fixture_handler(prompt, type("SynthesisOutput", (), {}))
            elif schema_name == "BaselineModelOutput":
                payload = fixture_handler(prompt, type("BaselineModelOutput", (), {}))
            elif schema_name == "BaselineCritiqueOutput":
                payload = fixture_handler(prompt, type("BaselineCritiqueOutput", (), {}))
            else:
                raise AssertionError(f"Unexpected OpenAI schema: {schema_name}")
            return SimpleNamespace(
                id="resp_mock_123",
                status="completed",
                incomplete_details=None,
                model="gpt-6-luna",
                output_text=json.dumps(payload),
                output=[],
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=10),
                ),
            )

    class _FakeAsyncOpenAI:
        def __init__(self) -> None:
            self.responses = _FakeResponses()

        async def close(self) -> None:
            return None

    db_path = tmp_path / "prod_openai.db"
    budget_path = tmp_path / "prod_openai_budget.db"
    # Note: NO explicit pricing_json — relies on DEFAULT_MODEL_PRICING for gpt-6-luna!
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="gpt-6-luna",
        openai_api_key="sk-mocked-only-never-sent",
        cloud_reasoning_profile=True,
        allow_paid_runs=True,
        max_cost_per_run_usd=0.05,
        max_cumulative_cost_usd=1.0,
        protected_reserve_usd=0.0,
        db_path=db_path,
        budget_authority_path=budget_path,
    )
    assert settings.effective_max_workers > 3  # adaptive by default

    mgr = PublicJobManager(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_path,
        provider_factory=lambda s: OpenAIProvider(
            model=s.model,
            api_key="sk-mocked-only-never-sent",
            client=_FakeAsyncOpenAI(),  # type: ignore[arg-type]
            reasoning_effort=s.openai_reasoning_effort,
            provider_label=s.provider.value,
        ),
    )

    net_sub = await mgr.enqueue_task(
        prompt="Verify the documented budget ceiling.",
        mode="network",
    )
    base_sub = await mgr.enqueue_task(
        prompt="Verify the documented budget ceiling.",
        mode="baseline",
    )
    for tid in (net_sub["task_id"], base_sub["task_id"]):
        for _ in range(50):
            st = await mgr.get_task_status(tid)
            if st["status"] in {"completed", "partial", "failed"}:
                break
            await asyncio.sleep(0.05)

    net_res = await mgr.get_task_result(net_sub["task_id"])
    base_res = await mgr.get_task_result(base_sub["task_id"])
    net_trace = await mgr.get_task_trace(net_sub["task_id"])

    assert net_res["status"] in {"completed", "partial"}, net_res.get("error")
    assert net_res["answer"]
    assert net_trace["found"] is True
    assert len(net_trace["agents"]) == 1
    assert base_res["status"] == "completed", base_res.get("error")
    assert "420 USD" in base_res["answer"]
    assert base_res["verification"]["independent_verification_performed"] is False
    await mgr.shutdown()


async def test_public_mcp_idempotency_stable_with_documents_failures_and_restarts(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "idem_docs.db"
    budget_path = tmp_path / "idem_budget.db"
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-public-model",
        openai_api_key="sk-offline",
        db_path=db_path,
        budget_authority_path=budget_path,
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
    )
    provider_calls = 0

    def _counting_handler(prompt: str, response_model: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        return _scripted_handler(prompt, response_model)

    mgr = PublicJobManager(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_path,
        provider_factory=lambda s: FakeProvider(
            handler=_counting_handler,
            provider_name="fake-public",
            model_name=s.model,
            delay_seconds=0.05,
        ),
    )

    docs = [{"title": "Policy", "content": "Budget cap is 420 USD."}]
    first = await mgr.enqueue_task(
        prompt="Verify policy.",
        reference_documents=docs,
        mode="network",
        idempotency_key="doc-idem-1",
    )
    assert first["accepted"] is True
    assert first["deduplicated"] is False

    # Wait briefly so retrieved_at timestamp would differ, then submit identical docs
    await asyncio.sleep(0.02)
    second = await mgr.enqueue_task(
        prompt="Verify policy.",
        reference_documents=[{"title": "Policy", "content": "Budget cap is 420 USD."}],
        mode="network",
        idempotency_key="doc-idem-1",
    )
    assert second["accepted"] is True
    assert second["deduplicated"] is True
    assert second["task_id"] == first["task_id"]

    # Changed document content with same idempotency_key must conflict
    changed = await mgr.enqueue_task(
        prompt="Verify policy.",
        reference_documents=[{"title": "Policy", "content": "Budget cap is 999 USD."}],
        mode="network",
        idempotency_key="doc-idem-1",
    )
    assert changed["accepted"] is False
    assert changed["error_code"] == "idempotency_conflict"

    for _ in range(40):
        st = await mgr.get_task_status(first["task_id"])
        if st["status"] in {"completed", "partial", "failed"}:
            break
        await asyncio.sleep(0.05)
    calls_after_first = provider_calls
    await mgr.shutdown()

    # Simulate a failed/interrupted job in SQLite (including legacy hash with retrieved_at)
    legacy_docs_json = json.dumps(
        [
            {
                "doc_id": "doc_1",
                "title": "Doc",
                "content": "Legacy content",
                "source_ref": "doc_1",
                "reliability": "medium",
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO public_jobs (
                task_id, mode, status, prompt, reference_documents_json,
                idempotency_key, request_hash,
                current_stage, progress, provider, model,
                result_json, error, created_at, updated_at, completed_at
            ) VALUES (
                'tsk_pub_legacy_running', 'network', 'running', 'Legacy prompt',
                ?, 'legacy-idem-1', 'old_hash_with_retrieved_at',
                'specialist_execution', 0.4, 'openai', 'fake-public-model',
                NULL, NULL, '2026-11-27T00:00:00Z', '2026-11-27T00:00:00Z', NULL
            )
            """,
            (legacy_docs_json,),
        )
        conn.commit()
    finally:
        conn.close()

    # Restart marks interrupted job failed; repeated request returns existing failed task
    # without triggering duplicate inference!
    mgr_restarted = PublicJobManager(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_path,
        provider_factory=lambda s: FakeProvider(
            handler=_counting_handler,
            provider_name="fake-public",
            model_name=s.model,
        ),
    )
    retry_after_restart = await mgr_restarted.enqueue_task(
        prompt="Legacy prompt",
        reference_documents=[{"title": "Doc", "content": "Legacy content"}],
        mode="network",
        idempotency_key="legacy-idem-1",
    )
    assert retry_after_restart["accepted"] is True
    assert retry_after_restart["deduplicated"] is True
    assert retry_after_restart["task_id"] == "tsk_pub_legacy_running"
    assert retry_after_restart["status"] == "failed"
    assert provider_calls == calls_after_first
    await mgr_restarted.shutdown()


async def test_public_mcp_budget_calculation_multidoc_provenance_trace_and_baseline(
    tmp_path: Path,
) -> None:
    """Reproduce budget calculation ($10,800 spent, $1,200 remaining, within budget)."""
    from crewforge.schemas import (
        ArithmeticRequest,
        Claim,
        ClaimCategory,
        ClaimStatus,
        Evidence,
        RunConfig,
        ToolExecutionRecord,
    )
    from crewforge.storage.database import Database
    from crewforge.tools.arithmetic import (
        DeterministicArithmeticTool,
        build_provenance_catalog,
    )

    # 1. Provenance checks: doc_1/doc_2, no collision, reject unverified/stale/cross-run
    ev_doc1 = Evidence(
        evidence_id="evd_doc1",
        run_id="run_a",
        source_type="external_retrieved",
        source_ref="doc_1",
        excerpt="Approved Budget Cap: 12000 USD",
        created_by="lead",
        verified_in_input=True,
    )
    ev_doc2 = Evidence(
        evidence_id="evd_doc2",
        run_id="run_a",
        source_type="external_retrieved",
        source_ref="doc_2",
        excerpt="Compute 4500, Storage 2300, Staffing 4000",
        created_by="lead",
        verified_in_input=True,
    )
    ev_specialist_collision = Evidence(
        evidence_id="evd_agt1",
        run_id="run_a",
        source_type="external_retrieved",
        source_ref="doc_1",
        excerpt="Malicious overwrite 999999",
        created_by="agt_1",
        verified_in_input=True,
    )
    ev_unverified = Evidence(
        evidence_id="evd_unv",
        run_id="run_a",
        source_type="external_retrieved",
        source_ref="doc_3",
        excerpt="Unverified 777",
        created_by="lead",
        verified_in_input=False,
    )
    stale_v1 = Claim(
        claim_id="clm_stale1",
        run_id="run_a",
        version=1,
        statement="Old value 500",
        category=ClaimCategory.COMPUTATION,
        status=ClaimStatus.SUPPORTED,
        author_agent_id="agt_1",
    )
    fresh_v2 = Claim(
        claim_id="clm_stale1",
        run_id="run_a",
        version=2,
        statement="New value 600",
        category=ClaimCategory.COMPUTATION,
        status=ClaimStatus.SUPPORTED,
        author_agent_id="agt_1",
    )
    cross_run = Claim(
        claim_id="clm_other_run",
        run_id="run_other",
        version=1,
        statement="Cross run 888",
        category=ClaimCategory.COMPUTATION,
        status=ClaimStatus.SUPPORTED,
        author_agent_id="agt_1",
    )
    cross_run_tool = ToolExecutionRecord(
        tool_execution_id="tlx_cross_run",
        run_id="run_other",
        actor="agt_1",
        expression="888",
        computed_result="888",
        units="USD",
        inputs_grounded=True,
        success=True,
    )
    cat = build_provenance_catalog(
        original_prompt="Check budget",
        verified_evidence=[ev_doc1, ev_doc2, ev_specialist_collision, ev_unverified],
        all_claims=[stale_v1, fresh_v2, cross_run],
        tool_records=[cross_run_tool],
        run_id="run_a",
    )
    assert "tlx_cross_run" not in cat.targets_by_key
    assert "doc_1" in cat.targets_by_key
    assert "doc_2" in cat.targets_by_key
    assert "12000" in cat.targets_by_key["doc_1"].text
    assert "999999" not in cat.targets_by_key["doc_1"].text

    prov_db = Database(tmp_path / "prov_sec.db")
    await prov_db.initialize()
    await prov_db.create_run(
        run_id="run_a",
        mode="network",
        prompt="Check budget",
        config=RunConfig(
            provider="fake",
            model="fake-model",
            request_timeout_seconds=10.0,
            max_retries=0,
            max_concurrency=1,
            max_output_tokens=1024,
            max_model_calls_per_run=8,
            max_cost_per_run_usd=0.0,
            max_cumulative_cost_usd=0.0,
        ),
    )
    arith_tool = DeterministicArithmeticTool(prov_db)

    for bad_ref, bad_val in (
        ("doc_3", "777"),
        ("clm_fabricated", "100"),
        ("clm_other_run", "888"),
        ("clm_stale1@v1", "500"),
        ("tlx_cross_run", "888"),
    ):
        rec = await arith_tool.execute(
            run_id="run_a",
            actor="lead",
            request=ArithmeticRequest.model_validate(
                {
                    "expression": "x",
                    "variable_bindings": [
                        {"name": "x", "value": bad_val, "provenance_ref": bad_ref}
                    ],
                    "proposed_result": bad_val,
                }
            ),
            catalog=cat,
        )
        assert rec.inputs_grounded is False or rec.success is False, (
            f"Expected rejection for {bad_ref}"
        )
    await prov_db.close()

    # 2. End-to-end public MCP network + baseline execution with doc_1 & doc_2
    def _budget_handler(prompt: str, response_model: Any) -> Any:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Compute total expenditure, remaining budget, and budget compliance",
                "user_requirements": [
                    "Calculate total expenditure",
                    "Calculate remaining budget and confirm within budget",
                ],
                "acceptance_criteria": ["Verified dollar figures grounded in doc_1 and doc_2"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Financial Analyst",
                        "specialty": "Budgeting",
                        "instructions": "Compute expenditure and remaining balance.",
                        "capabilities": ["reasoning"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute budget totals",
                        "description": "Sum expenses from doc_2 and compare against doc_1 cap.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": [
                            "Expenditure 10800, Remaining 1200, Within budget"
                        ],
                    }
                ],
                "assumptions": [],
                "missing_information": [],
            }
        if schema_name in {"WorkerOutput", "SpecialistOutput"}:
            return {
                "findings": (
                    "Total expenditure is $10,800, remaining budget is $1,200, "
                    "and spending is within budget."
                ),
                "evidence": [
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_1",
                        "excerpt": "Approved Budget Cap: 12000 USD",
                    },
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_2",
                        "excerpt": "Compute 4500 USD, Storage 2300 USD, Staffing 4000 USD",
                    },
                ],
                "claims": [
                    {
                        "statement": "Total expenditure is $10,800.",
                        "category": "computation",
                        "evidence_indices": [1],
                        "requirement_refs": ["req_1"],
                        "computation": {
                            "expression": "compute + storage + staffing",
                            "variable_bindings": [
                                {"name": "compute", "value": "4500", "provenance_ref": "doc_2"},
                                {"name": "storage", "value": "2300", "provenance_ref": "doc_2"},
                                {"name": "staffing", "value": "4000", "provenance_ref": "doc_2"},
                            ],
                            "proposed_result": "10800",
                            "units": "USD",
                        },
                    },
                    {
                        "statement": "Remaining budget is $1,200, which is within budget.",
                        "category": "computation",
                        "evidence_indices": [0, 1],
                        "requirement_refs": ["req_2"],
                        "computation": {
                            "expression": "cap - (compute + storage + staffing)",
                            "variable_bindings": [
                                {"name": "cap", "value": "12000", "provenance_ref": "doc_1"},
                                {"name": "compute", "value": "4500", "provenance_ref": "doc_2"},
                                {"name": "storage", "value": "2300", "provenance_ref": "doc_2"},
                                {"name": "staffing", "value": "4000", "provenance_ref": "doc_2"},
                            ],
                            "proposed_result": "1200",
                            "units": "USD",
                        },
                    },
                ],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
            }
        if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
            cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
            return {
                "verdicts": [
                    {
                        "claim_id": cid,
                        "decision": "approve",
                        "rationale": "Verified via deterministic arithmetic against doc_1/doc_2.",
                        "evidence_refs": [],
                        "required_correction": None,
                        "blocking_severity": "none",
                        "blocking_issues": [],
                    }
                    for cid in cids
                ]
            }
        if schema_name == "SynthesisOutput":
            cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
            refs = [{"claim_id": cid, "claim_version": 1} for cid in cids]
            return {
                "direct_summary": (
                    "Total expenditure is $10,800, leaving $1,200 remaining (within budget)."
                ),
                "accepted_segments": [
                    {
                        "segment_type": "accepted_claim",
                        "text": "Total expenditure is $10,800.",
                        "claim_refs": refs[:1],
                        "requirement_ref": "req_1",
                    },
                    {
                        "segment_type": "accepted_claim",
                        "text": "Remaining budget is $1,200, which is within budget.",
                        "claim_refs": refs[1:2] or refs[:1],
                        "requirement_ref": "req_2",
                    },
                ],
                "conditional_segments": [],
                "recommendations": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": cids,
                "assumptions": [],
                "missing_information": [],
            }
        if schema_name == "BaselineModelOutput":
            return {
                "answer": (
                    "Baseline: Total expenditure is $10,800, remaining budget is $1,200 "
                    "(within budget)."
                ),
                "missing_information": [],
                "assumptions": [],
                "computations": [],
                "confidence_notes": "Baseline calculation.",
            }
        if schema_name == "BaselineCritiqueOutput":
            return {
                "unanswered_requirements": [],
                "arithmetic_issues": [],
                "missing_context_or_assumptions": [],
                "is_fully_satisfied": True,
                "computations": [],
                "refined_answer": (
                    "Baseline: Total expenditure is $10,800, remaining budget is $1,200 "
                    "(within budget)."
                ),
                "missing_information": [],
                "assumptions": [],
                "confidence_notes": "Baseline verified.",
            }
        raise AssertionError(f"Unexpected schema: {schema_name}")

    db_path = tmp_path / "budget_repro.db"
    budget_path = tmp_path / "budget_repro_auth.db"
    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-budget-model",
        openai_api_key="sk-offline",
        db_path=db_path,
        budget_authority_path=budget_path,
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
    )
    mgr = PublicJobManager(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_path,
        provider_factory=lambda s: FakeProvider(
            handler=_budget_handler, provider_name="fake", model_name=s.model
        ),
    )
    docs = [
        {"title": "Q4 Budget Cap", "content": "Approved Budget Cap: 12000 USD"},
        {
            "title": "Q4 Line Items",
            "content": "Compute 4500 USD, Storage 2300 USD, Staffing 4000 USD",
        },
    ]
    net_sub = await mgr.enqueue_task(
        prompt="Compute total expenditure, remaining budget, and whether within budget.",
        reference_documents=docs,
        mode="network",
    )
    base_sub = await mgr.enqueue_task(
        prompt="Compute total expenditure, remaining budget, and whether within budget.",
        reference_documents=docs,
        mode="baseline",
    )
    for tid in (net_sub["task_id"], base_sub["task_id"]):
        for _ in range(50):
            st = await mgr.get_task_status(tid)
            if st["status"] in {"completed", "partial", "failed"}:
                break
            await asyncio.sleep(0.05)

    net_res = await mgr.get_task_result(net_sub["task_id"])
    assert net_res["status"] == "completed", net_res.get("error")
    assert net_res["verification"]["honesty_gate_passed"] is True
    assert "10,800" in net_res["answer"] or "10800" in net_res["answer"]
    assert "1,200" in net_res["answer"] or "1200" in net_res["answer"]
    assert "within budget" in net_res["answer"].lower()
    src_titles = {s.get("title") for s in net_res["sources"] if s.get("title")}
    assert src_titles == {"Q4 Budget Cap", "Q4 Line Items"}

    net_trace = await mgr.get_task_trace(net_sub["task_id"])
    assert net_trace["found"] is True
    assert len(net_trace["tool_executions"]) == 2
    assert all(t["success"] is True for t in net_trace["tool_executions"])
    assert {t["computed_result"] for t in net_trace["tool_executions"]} == {"10800", "1200"}

    base_res = await mgr.get_task_result(base_sub["task_id"])
    assert base_res["status"] == "completed"
    assert "10,800" in base_res["answer"]
    assert "1,200" in base_res["answer"]
    assert base_res["verification"]["independent_verification_performed"] is False
    await mgr.shutdown()




