# Single-replica production container with persistent /data SQLite storage
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

USER crewforge

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=3)"

CMD ["sh", "-c", "exec crewforge-public-mcp --host 0.0.0.0 --port ${PORT:-8000}"]
