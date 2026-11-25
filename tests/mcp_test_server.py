"""Subprocess entrypoint launching CrewForge MCP server with FakeProvider for offline tests."""

from __future__ import annotations

import os
from pathlib import Path

from crewforge.config import CrewForgeSettings
from crewforge.mcp_server import create_mcp_server
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider


def main() -> None:
    db_path = Path(os.environ["CREWFORGE_DB_PATH"])
    settings = CrewForgeSettings(db_path=db_path)
    database = Database(db_path)
    provider = FakeProvider(
        provider_name="fake-mcp",
        model_name="fake-mcp-model",
        is_paid_provider=False,
    )
    server = create_mcp_server(
        settings=settings,
        database=database,
        provider=provider,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
