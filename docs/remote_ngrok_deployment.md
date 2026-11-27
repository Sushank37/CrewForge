# CrewForge Public Deployment Guide (ngrok)

This guide describes how to expose CrewForge securely over HTTPS using **ngrok** at the assigned public hostname (`stairlike-causatively-cordell.ngrok-free.dev`) with **traffic inspection disabled**, while keeping local development accounts, local Ollama, and administrator endpoints strictly isolated to `127.0.0.1`.

## Security Controls Enforced for Public Deployment

- **Approved HTTPS Model APIs Only (SSRF Protection):**
  - Remote user and judge sessions (`usr_*`, `judge_*`) and remote `/api/settings` updates only accept `https://` endpoints on the approved model API allowlist (`api.openai.com`, `api.groq.com`, `generativelanguage.googleapis.com`, `api.anthropic.com`, `openrouter.ai`, `api.together.xyz`, `api.fireworks.ai`, `api.mistral.ai`, `api.deepseek.com`).
  - `localhost`, `*.local`, `*.internal`, raw IP addresses, loopback, private RFC 1918 ranges, and link-local/metadata addresses (`169.254.169.254`) are rejected.
- **Local Ollama Restricted to Local Development:**
  - `provider="ollama"` is permitted **only** on direct localhost requests for the local `default` development account. Remote sessions and remote requests cannot select or invoke local Ollama.
- **Traffic Inspection Disabled (`--inspect=false`):**
  - Disabling ngrok's local web inspection interface ensures BYOK API keys, HttpOnly session cookies, CSRF tokens, email OTP codes, and user prompts are never recorded in ngrok's HTTP inspector (`127.0.0.1:4040`).
- **Host Header & Forwarding Protection:**
  - Requests with unrecognized `Host` headers are rejected (`403`).
  - Requests carrying `X-Forwarded-For`, `X-Forwarded-Host`, `Forwarded`, or `CF-Connecting-IP` are treated as remote, blocking `/api/admin/*` and `CREWFORGE_ADMIN_TOKEN` even if the tunnel rewrites the `Host` header.
- **Per-User & MCP Isolation:**
  - Every authenticated user/judge receives isolated SQLite trace/budget databases, isolated in-memory BYOK credentials, HttpOnly+Secure+SameSite cookies with CSRF protection, and server-side MCP session ownership (`mcp-session-id` binding).

---

## 1. Start CrewForge Locally with Host Allowlisting

Bind CrewForge strictly to `127.0.0.1:8765` and allowlist the public ngrok hostname:

```bash
export CREWFORGE_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export CREWFORGE_ALLOWED_HOSTS="stairlike-causatively-cordell.ngrok-free.dev"

uv run crewforge-dashboard \
  --host 127.0.0.1 \
  --port 8765 \
  --allowed-host stairlike-causatively-cordell.ngrok-free.dev
```

## 2. Start ngrok with Traffic Inspection Disabled

Run ngrok pointing at `127.0.0.1:8765` with `--inspect=false`:

```bash
ngrok http 127.0.0.1:8765 \
  --domain=stairlike-causatively-cordell.ngrok-free.dev \
  --inspect=false
```

Or if using an `ngrok.yml` configuration file, set `inspect: false`:

```yaml
version: "2"
tunnels:
  crewforge:
    proto: http
    addr: 127.0.0.1:8765
    domain: stairlike-causatively-cordell.ngrok-free.dev
    inspect: false
```

Start with:

```bash
ngrok start crewforge
```

## 3. Remote Client & MCP Usage

- **Web UI / Auth:** `https://stairlike-causatively-cordell.ngrok-free.dev`
- **Remote Streamable HTTP MCP Endpoint:** `https://stairlike-causatively-cordell.ngrok-free.dev/mcp`
  - Authenticate with `Authorization: Bearer <USER_OR_JUDGE_MCP_TOKEN>`
  - Configure BYOK settings via `POST https://stairlike-causatively-cordell.ngrok-free.dev/api/settings` using an approved HTTPS model API (e.g., `https://api.openai.com/v1` or `https://api.groq.com/openai/v1`).
