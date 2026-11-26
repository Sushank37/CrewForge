import React, { useEffect, useState } from "react";
import { api } from "../api";
import type { McpStatus } from "../types";

export const McpView: React.FC = () => {
  const [status, setStatus] = useState<McpStatus | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = () => {
    setErr(null);
    api
      .getMcpStatus()
      .then(setStatus)
      .catch((e) => setErr(e.message));
  };

  useEffect(() => {
    load();
  }, []);

  const copyText = (label: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 1800);
  };

  if (!status) {
    return (
      <div className="page-container">
        {err ? <div className="alert-error">{err}</div> : <div className="muted">Checking MCP status...</div>}
      </div>
    );
  }

  const stdioJson = JSON.stringify(status.stdio_config, null, 2);

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1>MCP Connection</h1>
          <p className="muted">
            CrewForge exposes <code>solve</code>, <code>baseline</code>, and <code>trace</code> over
            stdio and localhost Streamable HTTP using the exact same execution engine and SQLite database.
          </p>
        </div>
        <button type="button" className="btn-ghost btn-sm" onClick={load}>
          Refresh Status
        </button>
      </div>

      <div className="settings-grid">
        <section className="card">
          <h2>Connection Status</h2>
          <div className="status-line">
            <span>Streamable HTTP Handshake (<code>{status.http_endpoint}</code>)</span>
            <span className={`pill ${status.http_reachable ? "pill-green" : "pill-amber"}`}>
              {status.http_reachable ? "Handshake OK" : "Not Reachable"}
            </span>
          </div>
          <div className="muted" style={{ fontSize: 12 }}>
            Shared session DB: <code>{status.session_db_path}</code>
            <br />
            Discovered tools: {(status.discovered_tools?.length ? status.discovered_tools : status.tools).map((t) => <code key={t} style={{ marginRight: 6 }}>{t}</code>)}
            {status.handshake_error && (
              <>
                <br />
                <span style={{ color: "var(--amber)" }}>Handshake note: {status.handshake_error}</span>
              </>
            )}
          </div>
          <div className="alert-info">
            Localhost-only by default. Do not expose an unauthenticated MCP server on a public tunnel.
          </div>
        </section>

        <section className="card">
          <div className="card-header-row">
            <h2>Stdio MCP Client Config</h2>
            <button
              type="button"
              className="btn-ghost btn-sm"
              onClick={() => copyText("stdio", stdioJson)}
            >
              {copied === "stdio" ? "Copied!" : "Copy JSON"}
            </button>
          </div>
          <pre className="code-pre">{stdioJson}</pre>
        </section>

        <section className="card">
          <div className="card-header-row">
            <h2>Local Streamable HTTP Command</h2>
            <button
              type="button"
              className="btn-ghost btn-sm"
              onClick={() => copyText("http", status.http_command)}
            >
              {copied === "http" ? "Copied!" : "Copy Command"}
            </button>
          </div>
          <pre className="code-pre">{status.http_command}</pre>
        </section>
      </div>
    </div>
  );
};
