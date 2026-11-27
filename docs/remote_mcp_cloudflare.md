# CrewForge Remote HTTPS MCP Deployment (Cloudflare Tunnel)

This guide explains how to expose CrewForge's authenticated Streamable HTTP MCP endpoint (`/mcp`) and BYOK configuration API (`/api/settings`) over HTTPS at zero cost using **Cloudflare Tunnel (`cloudflared`)**, while keeping the local administrator interface and development session strictly restricted to `127.0.0.1`.

## Security Model

- **Local Administrator (`default` session & `/api/admin/*`)**:
  - Uses `CREWFORGE_ADMIN_TOKEN` (generated automatically if unset).
  - Accepted **only** on direct localhost requests (`127.0.0.1` / `localhost` without tunnel forwarding headers). Rejected (`403`) if presented over a remote tunnel.
- **Remote Judges (`judge_*` isolated sessions)**:
  - Each judge receives a unique, revocable Bearer token (`cfj_...`) issued by the local administrator.
  - Session identity is derived **server-side** from the SHA-256 hash of the presented Bearer token.
  - Each judge gets isolated SQLite databases (`session.db` and `budget.db`), isolated in-memory BYOK credentials, and isolated MCP Streamable HTTP sessions (`mcp-session-id` ownership enforced).
  - Judges cannot access `/api/admin/*`, other judges' sessions, or the `default` development session.
- **No Token or Credential Leakage**:
  - Raw authentication tokens and provider API keys are never embedded in HTML, logs, `/api/mcp/status`, or `/api/admin/tokens` listings.

---

## 1. Start CrewForge Locally with Tunnel Host Allowlisting

Set a strong local administrator token and allow only your Cloudflare Tunnel hostname:

```bash
export CREWFORGE_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export CREWFORGE_ALLOWED_HOSTS="your-tunnel-hostname.trycloudflare.com"

uv run crewforge-dashboard \
  --host 127.0.0.1 \
  --port 8765 \
  --allowed-host your-tunnel-hostname.trycloudflare.com
```

> Note: CrewForge stays bound to `127.0.0.1:8765`. Only `cloudflared` forwards HTTPS traffic to it.

## 2. Start Cloudflare Tunnel (`cloudflared`)

### Option A: Zero-Cost Quick Tunnel (ephemeral URL, no account required)

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

Copy the generated `https://<random>.trycloudflare.com` hostname and pass it via `--allowed-host <random>.trycloudflare.com` (or `CREWFORGE_ALLOWED_HOSTS`).

### Option B: Named Cloudflare Tunnel (stable custom domain, free Cloudflare plan)

```bash
cloudflared tunnel login
cloudflared tunnel create crewforge-mcp
cloudflared tunnel route dns crewforge-mcp mcp.yourdomain.com
cloudflared tunnel run --url http://127.0.0.1:8765 crewforge-mcp
```

## 3. Issue & Revoke Per-Judge Tokens (Localhost Only)

From the host machine (`127.0.0.1`), issue a unique token for each judge:

```bash
curl -sS -X POST http://127.0.0.1:8765/api/admin/tokens \
  -H "Authorization: Bearer ${CREWFORGE_ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"label": "Judge Alice"}'
```

Example response (save `token` securely and send it to that judge; it is only returned once):

```json
{
  "token_id": "tok_a1b2c3d4e5f6",
  "session_id": "judge_9f8e7d6c5b4a",
  "label": "Judge Alice",
  "token": "cfj_...",
  "created_at": "2026-11-27T06:00:00+00:00"
}
```

List active/revoked judge tokens (non-secret metadata only):

```bash
curl -sS http://127.0.0.1:8765/api/admin/tokens \
  -H "Authorization: Bearer ${CREWFORGE_ADMIN_TOKEN}"
```

Revoke a judge token immediately (by `token_id` or `session_id`):

```bash
curl -sS -X DELETE http://127.0.0.1:8765/api/admin/tokens/judge_9f8e7d6c5b4a \
  -H "Authorization: Bearer ${CREWFORGE_ADMIN_TOKEN}"
```

## 4. Judge BYOK Configuration (Authenticated)

Each judge configures their own preferred model and API key in their isolated session (stored in RAM only, unrestricted by development quotas):

```bash
curl -sS -X POST https://your-tunnel-hostname.trycloudflare.com/api/settings \
  -H "Authorization: Bearer <JUDGE_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "billing_mode": "byok",
    "provider": "openai",
    "model": "gpt-5.4",
    "api_key": "<JUDGE_PROVIDER_API_KEY>",
    "allow_paid_runs": true,
    "max_output_tokens": 16384
  }'
```

## 5. Standard Remote MCP Client Configuration

Standard remote MCP clients (Claude Desktop, Cursor, MCP Inspector, or `mcp.client.streamable_http`) connect using only the HTTPS `/mcp` URL and the judge's Bearer token:

```json
{
  "mcpServers": {
    "crewforge": {
      "url": "https://your-tunnel-hostname.trycloudflare.com/mcp",
      "headers": {
        "Authorization": "Bearer <JUDGE_TOKEN>"
      }
    }
  }
}
```

Available MCP tools: `solve`, `baseline`, and `trace`.
