"""Verify the release entry point keeps the anonymous and private surfaces separate."""

import json
import sqlite3
import time

from starlette.testclient import TestClient

from crewforge.config import CrewForgeSettings
from crewforge.production import create_release_app
from crewforge.public_mcp_service import create_public_mcp_app
from tests.fake_provider import ChecklistFixtureProvider
from tests.test_public_mcp_service import _scripted_handler


def test_release_serves_dashboard_and_anonymous_mcp(monkeypatch, tmp_path):
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text('<script type="module" src="/assets/app.js"></script>')
    assets = frontend / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("void 0;")
    monkeypatch.setattr("crewforge.dashboard._FRONTEND_DIST_DIR", frontend)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "crewforge.example")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "a" * 40)
    monkeypatch.setenv("CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE", "false")
    monkeypatch.setenv("CREWFORGE_DASHBOARD_DATA_DIR", str(tmp_path / "dashboard"))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_DB_PATH", str(tmp_path / "public.db"))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH", str(tmp_path / "budget.db"))
    app = create_release_app(settings=CrewForgeSettings(
        _env_file=None, model="test-model", provider="openai", allow_paid_runs=True,
    ))
    headers = {"Accept": "application/json, text/event-stream"}
    with TestClient(app, base_url="https://crewforge.example") as client:
        assert client.get("/health").json()["commit"] == "a" * 40
        assert client.get("/health").json()["public_paid_inference_enabled"] is False
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").text == "void 0;"
        session = client.get("/api/guest/session")
        assert session.json()["available"] is False
        assert session.headers["cache-control"] == "no-store"
        assert {"HttpOnly", "Secure", "SameSite=strict", "Path=/api/guest"} <= {
            part.strip() for part in session.headers["set-cookie"].split(";")
        }
        guest_submit = client.post("/api/guest/jobs", json={"prompt": "Do this"}, headers={
            "Origin": "https://crewforge.example",
        })
        assert guest_submit.status_code == 503
        assert "paid inference is disabled" in guest_submit.json()["error"]
        assert client.get("/api/chats").status_code == 401
        assert client.get("/api/settings").status_code == 401
        assert client.post("/api/runs/some-run/compare").status_code == 401
        assert client.post("/mcp/workspace", headers=headers, json={}).status_code == 401
        init = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "offline", "version": "1.0"},
            },
        })
        assert init.status_code == 200, init.text
        headers["Mcp-Session-Id"] = init.headers["mcp-session-id"]
        client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        tools = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        assert tools.status_code == 200 and '"run_task"' in tools.text
        assert all(name in tools.text for name in (
            "run_single_agent", "get_task_status", "get_task_result",
            "get_task_trace", "cancel_task",
        ))
        assert "admin" not in tools.text


def test_guest_and_mcp_share_jobs_and_limits_without_sharing_job_access(monkeypatch, tmp_path):
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("CrewForge")
    monkeypatch.setattr("crewforge.dashboard._FRONTEND_DIST_DIR", frontend)
    monkeypatch.setenv("CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE", "true")
    monkeypatch.setenv("CREWFORGE_DASHBOARD_DATA_DIR", str(tmp_path / "dashboard"))
    job_db = tmp_path / "public.db"
    budget_db = tmp_path / "budget.db"
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_DB_PATH", str(job_db))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH", str(budget_db))
    providers = []

    def fake_factory(settings):
        provider = ChecklistFixtureProvider(
            handler=_scripted_handler, model_name=settings.model, delay_seconds=0.12,
        )
        providers.append(provider)
        return provider

    public_app = None

    def create_scripted_public_app(*, settings):
        nonlocal public_app
        public_app = create_public_mcp_app(
            settings=settings, provider_factory=fake_factory,
            max_concurrent_jobs=1, max_pending_jobs=1,
        )
        return public_app

    monkeypatch.setattr("crewforge.production.create_public_mcp_app", create_scripted_public_app)
    app = create_release_app(settings=CrewForgeSettings(
        _env_file=None, model="test-model", provider="openai", allow_paid_runs=True,
        max_cost_per_run_usd=0.0, max_cumulative_cost_usd=0.0,
    ))
    owner_origin = {"Origin": "https://testserver"}
    mcp_headers = {"Accept": "application/json, text/event-stream"}
    prompt = "Verify the budget ceiling. The budget ceiling is 420 USD."

    def wait_for_end(client, task_id):
        for _ in range(160):
            response = client.get(f"/api/guest/jobs/{task_id}")
            assert response.status_code == 200
            if response.json()["result"] is not None:
                return response.json()
            time.sleep(0.05)
        raise AssertionError("Guest job did not finish")

    with TestClient(app, base_url="https://testserver") as owner:
        stranger = TestClient(app, base_url="https://testserver")
        assert owner.get("/api/guest/session").json()["available"] is True
        assert stranger.get("/api/guest/session").json()["available"] is True
        assert owner.cookies["crewforge_guest"] != stranger.cookies["crewforge_guest"]
        assert owner.post("/api/guest/jobs", json={"prompt": prompt}).status_code == 403
        assert owner.post("/api/guest/jobs", json={"prompt": prompt}, headers={
            "Origin": "https://different.example",
        }).status_code == 403
        assert owner.post("/api/guest/jobs", json={
            "prompt": prompt, "api_key": "client-controlled",
        }, headers=owner_origin).status_code == 400
        assert owner.post("/api/guest/jobs", json={"prompt": "x" * 530_000},
                          headers=owner_origin).status_code == 413

        init = owner.post("/mcp", headers=mcp_headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "offline", "version": "1.0"},
            },
        })
        mcp_headers["Mcp-Session-Id"] = init.headers["mcp-session-id"]
        owner.post("/mcp", headers=mcp_headers, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })

        def call_mcp(name, **arguments):
            response = owner.post("/mcp", headers=mcp_headers, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })
            assert response.status_code == 200
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    return json.loads(line[6:])["result"]["structuredContent"]
            return response.json()["result"]["structuredContent"]

        created = owner.post("/api/guest/jobs", json={"prompt": prompt}, headers=owner_origin)
        assert created.status_code == 202
        team = created.json()["task_id"]
        assert call_mcp("run_task", prompt="Shared from MCP")["error_code"] == "capacity_exhausted"

        path = f"/api/guest/jobs/{team}"
        assert stranger.get(path).status_code == 404
        assert stranger.post(f"{path}/cancel", headers=owner_origin).status_code == 404
        assert stranger.post("/api/guest/jobs", json={"compare_to": team},
                             headers=owner_origin).status_code == 404
        for tool in ("get_task_status", "get_task_result", "get_task_trace", "cancel_task"):
            assert call_mcp(tool, task_id=team)["found"] is False
        assert owner.get("/api/chats").status_code == 401
        assert owner.get("/api/guest/jobs").status_code == 405
        team_result = wait_for_end(owner, team)
        assert team_result["trace"]["agents"]
        assert team_result["trace"]["review_decisions"]
        assert "420 USD" in team_result["result"]["answer"]
        assert team_result["result"]["usage"]["model_calls"] > 0

        compare = owner.post("/api/guest/jobs", json={"compare_to": team}, headers=owner_origin)
        assert compare.status_code == 202
        single = compare.json()["task_id"]
        single_result = wait_for_end(owner, single)
        assert single_result["result"]["answer"].startswith("Baseline:")
        assert single_result["result"]["verification"]["release_mode"] == "unverified_baseline"
        assert single_result["result"]["usage"]["model_calls"] > 0
        repeat = owner.post("/api/guest/jobs", json={"compare_to": team}, headers=owner_origin)
        assert repeat.json()["task_id"] == single
        assert len(providers) == 2
        assert public_app.state.job_manager.budget_db_path == budget_db
        with sqlite3.connect(job_db) as conn:
            rows = conn.execute(
                """SELECT mode, prompt, reference_documents_json, guest_owner FROM public_jobs
                   WHERE task_id IN (?, ?) ORDER BY mode""", (team, single),
            ).fetchall()
        assert [row[0] for row in rows] == ["baseline", "network"]
        assert rows[0][1:] == rows[1][1:]
        assert rows[0][1] == prompt and rows[0][3] != owner.cookies["crewforge_guest"]

        owned = owner.post(
            "/api/guest/jobs", json={"prompt": "Cancel this run"}, headers=owner_origin,
        )
        assert owned.status_code == 202
        cancel_path = f"/api/guest/jobs/{owned.json()['task_id']}/cancel"
        assert owner.post(cancel_path).status_code == 403
        assert owner.post(cancel_path, headers=owner_origin).json()["cancelled"] is True

        mcp_job = call_mcp("run_task", prompt="MCP-only task")["task_id"]
        assert owner.get(f"/api/guest/jobs/{mcp_job}").status_code == 404
        assert owner.post("/api/guest/jobs", json={"compare_to": mcp_job},
                          headers=owner_origin).status_code == 404
        assert call_mcp("cancel_task", task_id=mcp_job)["found"] is True


async def test_existing_public_jobs_remain_accessible_after_guest_schema_migration(tmp_path):
    db_path = tmp_path / "existing.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE public_jobs (
                task_id TEXT PRIMARY KEY, mode TEXT NOT NULL, status TEXT NOT NULL,
                prompt TEXT NOT NULL, reference_documents_json TEXT NOT NULL,
                idempotency_key TEXT, request_hash TEXT, current_stage TEXT NOT NULL,
                progress REAL NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                result_json TEXT, error TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, completed_at TEXT
            );
            CREATE UNIQUE INDEX idx_public_jobs_idempotency ON public_jobs (mode, idempotency_key)
                WHERE idempotency_key IS NOT NULL;
            INSERT INTO public_jobs VALUES (
                'preexisting', 'network', 'completed', 'Original prompt', '[]', 'keep-key', NULL,
                'completed', 1.0, 'openai', 'test-model', '{"answer":"Existing answer."}', NULL,
                '2026-01-01', '2026-01-01', '2026-01-01'
            );
        """)
    manager = create_public_mcp_app(
        settings=CrewForgeSettings(_env_file=None), db_path=db_path,
    ).state.job_manager
    assert (await manager.get_task_result("preexisting"))["answer"] == "Existing answer."
    assert not (await manager.get_task_result("preexisting", guest_owner="visitor"))["found"]
    repeat = await manager.enqueue_task(
        prompt="Original prompt", mode="network", idempotency_key="keep-key",
    )
    assert repeat["task_id"] == "preexisting"
    assert repeat["deduplicated"] is True


def test_guest_conversations_restore_and_compare_frozen_followup(monkeypatch, tmp_path):
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("CrewForge")
    monkeypatch.setattr("crewforge.dashboard._FRONTEND_DIST_DIR", frontend)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "crewforge.example")
    monkeypatch.setenv("CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE", "true")
    monkeypatch.setenv("CREWFORGE_DASHBOARD_DATA_DIR", str(tmp_path / "dashboard"))
    job_path = tmp_path / "public.db"
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_DB_PATH", str(job_path))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH", str(tmp_path / "budget.db"))
    calls = []

    def corrected_followup(prompt, schema):
        from crewforge.schemas import PlannerOutput, ReviewerBatchOutput, SpecialistOutput
        from tests.test_requirement_recovery import approved, one_task_plan

        statement = "The corrected hypothetical ceiling is 300 USD."
        if schema is PlannerOutput:
            return one_task_plan(["Apply the latest correction"], [
                ("req_1", "State the corrected hypothetical ceiling", "explanation"),
            ])
        if schema is SpecialistOutput:
            return {"findings": statement, "claims": [
                {"statement": statement, "category": "logical_inference",
                 "requirement_refs": ["deliv_1"]},
            ]}
        if schema is ReviewerBatchOutput:
            return approved(prompt, {
                statement: (["deliv_1"], "The current request explicitly overrides the old limit."),
            }, {})
        return _scripted_handler(prompt, schema)

    def scripted(*, settings):
        def provider_factory(_settings):
            provider = ChecklistFixtureProvider(
                handler=corrected_followup if len(calls) == 1 else _scripted_handler,
                model_name=settings.model,
            )
            calls.append(provider)
            return provider
        return create_public_mcp_app(settings=settings, provider_factory=provider_factory)

    monkeypatch.setattr("crewforge.production.create_public_mcp_app", scripted)
    settings = CrewForgeSettings(
        _env_file=None, provider="openai", model="test-model", allow_paid_runs=True,
        max_cost_per_run_usd=0.0, max_cumulative_cost_usd=0.0,
    )
    root = "https://crewforge.example"
    headers = {"Origin": root}
    client = TestClient(create_release_app(settings=settings), base_url=root)

    def send(c, chat, prompt, number, documents=None):
        return c.post(f"/api/guest/chats/{chat}/messages", headers=headers, json={
            "request_id": f"00000000-0000-0000-0000-{number:012d}", "prompt": prompt,
            "documents": documents or [],
        })

    def wait(c, task_id):
        for _ in range(150):
            data = c.get(f"/api/guest/jobs/{task_id}").json()
            if data["result"]:
                return data["result"]
            time.sleep(0.025)
        raise AssertionError("Guest message never completed")

    with client:
        client.get("/api/guest/session")
        chat = client.post("/api/guest/chats", headers=headers).json()["chat_id"]
        other_chat = client.post("/api/guest/chats", headers=headers).json()["chat_id"]
        assert not client.get(f"/api/guest/chats/{other_chat}").json()["turns"]

        stranger = TestClient(client.app, base_url=root)
        stranger.get("/api/guest/session")
        assert stranger.get("/api/guest/chats").json()["chats"] == []
        assert stranger.get(f"/api/guest/chats/{chat}").status_code == 404
        assert send(stranger, chat, "Cross-session injection", 1).status_code == 404
        assert client.get("/api/chats").status_code == 401

        docs = [{"title": "Budget", "content": "The budget ceiling is 420 USD."}]
        first_text = "Verify the reference budget ceiling."
        first = send(client, chat, first_text, 1, docs)
        assert first.status_code == 202
        first_id = first.json()["task_id"]
        assert "420 USD" in wait(client, first_id)["answer"]
        follow_text = "Correction: use a 300 USD ceiling for this hypothetical case."
        follow = send(client, chat, follow_text, 2)
        assert follow.status_code == 202
        second_id = follow.json()["task_id"]
        corrected = wait(client, second_id)["conversational_answer"]
        assert "300 USD" in corrected and "420 USD" not in corrected
        again = send(client, chat, follow_text, 2)
        assert again.json() == {"accepted": True, "task_id": second_id, "deduplicated": True}
        assert send(client, chat, "Different message", 2).status_code == 400
        assert "Historical conversation" in calls[1].calls[0]["prompt"]
        assert first_text in calls[1].calls[0]["prompt"]
        assert "420 USD" in calls[1].calls[0]["prompt"]
        assert "latest user correction supersedes" in calls[1].calls[0]["prompt"]
        assert follow_text in calls[1].calls[0]["prompt"]

        third = send(client, chat, "What if the limit changes again?", 3)
        assert third.status_code == 202
        wait(client, third.json()["task_id"])
        compare = client.post("/api/guest/jobs", headers=headers, json={"compare_to": second_id})
        single_id = compare.json()["task_id"]
        result = wait(client, single_id)
        assert result["verification"]["release_mode"] == "unverified_baseline"
        with sqlite3.connect(job_path) as conn:
            args = """SELECT prompt, reference_documents_json, conversation_context
                      FROM public_jobs WHERE task_id IN (?, ?)"""
            inputs = conn.execute(args, (second_id, single_id)).fetchall()
        assert inputs[0] == inputs[1]
        assert "What if the limit changes again?" not in inputs[0][2]
        assert '"title": "Budget"' in inputs[0][1]
        assert all("420 USD" not in doc["content"]
                   for doc in json.loads(inputs[0][1]) if doc["title"] != "Budget")

    with TestClient(create_release_app(settings=settings), base_url=root) as restored:
        restored.cookies.set("crewforge_guest", client.cookies["crewforge_guest"],
                             path="/api/guest")
        detail = restored.get(f"/api/guest/chats/{chat}").json()
        assert [t["prompt"] for t in detail["turns"]][:2] == [first_text, follow_text]
        assert detail["turns"][1]["comparison_id"] == single_id
        assert not restored.get(f"/api/guest/chats/{other_chat}").json()["turns"]


def test_guest_turn_cannot_spend_past_shared_production_cap(monkeypatch, tmp_path):
    from crewforge.config import ModelPricing
    from crewforge.core.budget import UnifiedBudgetAuthority

    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text("CrewForge")
    monkeypatch.setattr("crewforge.dashboard._FRONTEND_DIST_DIR", frontend)
    monkeypatch.delenv("RAILWAY_PUBLIC_DOMAIN", raising=False)
    monkeypatch.setenv("CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE", "true")
    monkeypatch.setenv("CREWFORGE_DASHBOARD_DATA_DIR", str(tmp_path / "dashboard"))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_DB_PATH", str(tmp_path / "public.db"))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH", str(tmp_path / "budget.db"))
    paid = ChecklistFixtureProvider(
        handler=_scripted_handler, provider_name="openai", model_name="test-model",
        is_paid_provider=True,
    )
    monkeypatch.setattr(
        "crewforge.production.create_public_mcp_app",
        lambda *, settings: create_public_mcp_app(
            settings=settings, provider_factory=lambda _: paid,
        ),
    )
    settings = CrewForgeSettings(
        _env_file=None, provider="openai", model="test-model", allow_paid_runs=True,
        max_cost_per_run_usd=0.05, max_cumulative_cost_usd=0.40,
        total_budget_ceiling_usd=5.90, protected_reserve_usd=5.50,
        pricing_json={"test-model": ModelPricing(input_per_1m=10, output_per_1m=20)},
    )
    authority = UnifiedBudgetAuthority(
        settings=settings.model_copy(update={"budget_authority_path": tmp_path / "budget.db"}),
        migrate_default_dbs=False,
    )
    with sqlite3.connect(authority.authority_path) as conn:
        conn.execute(
            """INSERT INTO canonical_budget_ledger (
                reservation_id, run_id, provider, model, status, reserved_cost_usd,
                actual_cost_usd, created_at, updated_at
            ) VALUES ('prior', 'prior', 'openai', 'test-model', 'reconciled',
                      0.399999, 0.399999, '2026-01-01', '2026-01-01')"""
        )
    root = "https://testserver"
    headers = {"Origin": root}
    with TestClient(create_release_app(settings=settings), base_url=root) as client:
        assert client.get("/api/guest/session").json()["available"]
        chat = client.post("/api/guest/chats", headers=headers).json()["chat_id"]
        job = client.post(f"/api/guest/chats/{chat}/messages", headers=headers, json={
            "request_id": "00000000-0000-0000-0000-000000000001",
            "prompt": "Verify a number.", "documents": [],
        })
        assert job.status_code == 202
        task = job.json()["task_id"]
        for _ in range(150):
            result = client.get(f"/api/guest/jobs/{task}").json()["result"]
            if result:
                break
            time.sleep(0.025)
        assert result and result["status"] in {"failed", "partial"}
        assert not result["verification"]["answer_complete"]
        assert not paid.calls


async def test_private_chat_uses_immutable_context_and_isolates_other_chats(tmp_path):
    import asyncio

    from crewforge.web_api import ChatMessageRequest, JudgeSessionManager

    calls = []

    def factory(settings):
        provider = ChecklistFixtureProvider(
            handler=_scripted_handler, model_name=settings.model,
        )
        calls.append(provider)
        return provider

    manager = JudgeSessionManager(
        db_path=tmp_path / "private.db", budget_db_path=tmp_path / "budget.db",
        provider_factory=factory,
    )
    chat = manager.create_chat()["chat_id"]
    isolated = manager.create_chat()["chat_id"]
    original = ChatMessageRequest(
        prompt="Check this ceiling.",
        attachments=[{"name": "Memo", "content": "The budget ceiling is 420 USD."}],
        request_id="00000000-0000-0000-0000-000000000001",
    )
    one = await manager.submit_message(chat, original)
    await asyncio.wait_for(manager._tasks[one["run_id"]], timeout=5)
    assert (await manager.submit_message(chat, original))["run_id"] == one["run_id"]
    next_request = ChatMessageRequest(prompt="Explain why the limit applies.")
    two = await manager.submit_message(chat, next_request)
    await asyncio.wait_for(manager._tasks[two["run_id"]], timeout=5)
    assert "Check this ceiling." in calls[1].calls[0]["prompt"]
    assert "The budget ceiling is 420 USD." in " ".join(call["prompt"] for call in calls[1].calls)
    assert not manager.get_chat(isolated)["messages"]
    comparison = await manager.compare_chat_run(two["run_id"])
    await asyncio.wait_for(manager._tasks[comparison["run_id"]], timeout=5)
    assert (await manager.compare_chat_run(two["run_id"])) == comparison
    assert len(manager.get_chat(chat)["messages"]) == 4
    assert manager.get_chat(chat)["messages"][-1]["comparison_run_id"] == comparison["run_id"]
    assert "Check this ceiling." in calls[2].calls[0]["prompt"]
