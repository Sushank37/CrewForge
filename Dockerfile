# Single-replica production container with persistent /data SQLite storage
# Note: Do not use Docker VOLUME here; configure a Railway Volume mounted at /data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    CREWFORGE_PUBLIC_MCP_DB_PATH=/data/public_mcp.db \
    CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH=/data/public_mcp_budget.db \
    CREWFORGE_PUBLIC_MAX_CONCURRENT_JOBS=4 \
    CREWFORGE_PUBLIC_MAX_PENDING_JOBS=12 \
    CREWFORGE_ALLOW_PAID_RUNS=false

WORKDIR /app

RUN useradd --create-home --uid 10001 crewforge \
    && mkdir -p /data \
    && chown -R crewforge:crewforge /app /data

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=3)"

CMD ["python", "-c", "import os, pathlib, pwd; pw = pwd.getpwnam('crewforge'); data_dir = pathlib.Path(os.environ.get('CREWFORGE_PUBLIC_MCP_DB_PATH', '/data/public_mcp.db')).parent; data_dir.mkdir(parents=True, exist_ok=True); (os.chown(data_dir, pw.pw_uid, pw.pw_gid), [os.chown(p, pw.pw_uid, pw.pw_gid) for p in data_dir.iterdir()], os.setgid(pw.pw_gid), os.setuid(pw.pw_uid)) if os.geteuid() == 0 else None; os.execvp('crewforge-public-mcp', ['crewforge-public-mcp', '--host', '0.0.0.0', '--port', os.environ.get('PORT', '8000')])"]
