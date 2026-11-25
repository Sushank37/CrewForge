"""Pytest fixtures enforcing complete isolation from production SQLite DBs and .env."""

from __future__ import annotations

from pathlib import Path

import pytest

from crewforge.core.budget import UnifiedBudgetAuthority


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all default DB/authority paths to tmp_path and strip real credentials."""
    UnifiedBudgetAuthority.reset_instances()

    test_state_dir = tmp_path / "crewforge_test_state"
    test_state_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("CREWFORGE_BUDGET_AUTHORITY_PATH", str(test_state_dir / "authority.db"))
    monkeypatch.setenv("CREWFORGE_ALLOW_PAID_RUNS", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CREWFORGE_OPENAI_API_KEY", raising=False)

    yield

    UnifiedBudgetAuthority.reset_instances()
