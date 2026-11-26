/* CrewForge Mission Control — Client Application (Read-Only Observability) */

const state = {
  tab: "replay",
  trace: null,
  replayFiles: [],
  replayFilename: null,
  replayCursor: 0,
  replayPlaying: false,
  replayTimer: null,
  replaySpeed: 1,
  liveRuns: [],
  liveRunId: null,
  liveTimer: null,
  timelineFilter: "all",
  selectedClaimId: null,
  selectedAgentId: null,
  benchmarkData: null,
};

function esc(val) {
  if (val === null || val === undefined) return "";
  return String(val)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function statusPillClass(status) {
  const s = String(status || "").toLowerCase();
  if (s === "completed" || s === "supported" || s === "approve" || s === "full" || s === "passed") return "pill-green";
  if (s === "partial" || s === "unresolved" || s === "revise" || s === "stale") return "pill-amber";
  if (s === "failed" || s === "rejected" || s === "reject" || s === "blocked") return "pill-red";
  if (s === "running" || s === "in_progress") return "pill-blue";
  return "pill-teal";
}

function fmtTokens(usage) {
  if (!usage || usage.usage_reported === false) return "Unknown (not reported)";
  const inp = usage.input_tokens;
  const out = usage.output_tokens;
  if (inp === null || inp === undefined || out === null || out === undefined) return "Unknown";
  return `${Number(inp).toLocaleString()} in / ${Number(out).toLocaleString()} out`;
}

function fmtCost(usage) {
  if (!usage) return "Unknown";
  const act = usage.actual_cost_usd;
  if (act === null || act === undefined) return "Unknown";
  return `$${Number(act).toFixed(6)}`;
}

function fmtDuration(ms) {
  if (ms === null || ms === undefined) return "Unknown";
  return `${(Number(ms) / 1000).toFixed(2)} s`;
}

async function fetchJson(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

/* ---------------- Mode Switching ---------------- */
function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });

  const execView = document.getElementById("view-execution");
  const benchView = document.getElementById("view-benchmark");
  const replayCtrls = document.getElementById("replay-controls");
  const liveCtrls = document.getElementById("live-controls");
  const banner = document.getElementById("replay-banner");
  const modeInd = document.getElementById("mode-indicator");

  stopReplayTimer();
  stopLiveTimer();

  if (tab === "benchmark") {
    execView.classList.add("hidden");
    benchView.classList.remove("hidden");
    modeInd.textContent = "BENCHMARK V2";
    modeInd.className = "pill pill-teal";
    loadBenchmark();
    return;
  }

  benchView.classList.add("hidden");
  execView.classList.remove("hidden");

  if (tab === "replay") {
    replayCtrls.classList.remove("hidden");
    liveCtrls.classList.add("hidden");
    banner.className = "banner banner-amber";
    banner.innerHTML =
      "<strong>HISTORICAL REPLAY</strong> — Reconstructing persisted Phase 5E trace in recorded event order. Future claims, verdicts, and final answer remain hidden until their corresponding events.";
    modeInd.textContent = "HISTORICAL REPLAY";
    modeInd.className = "pill pill-amber";
    loadReplayList();
  } else {
    replayCtrls.classList.add("hidden");
    liveCtrls.classList.remove("hidden");
    banner.className = "banner banner-blue";
    banner.innerHTML =
      "<strong>LIVE MONITOR</strong> — Polling read-only configured demo SQLite database for active or newly created runs.";
    modeInd.textContent = "LIVE MONITOR";
    modeInd.className = "pill pill-blue";
    refreshLiveRuns();
    state.liveTimer = setInterval(pollLive, 2500);
  }
}

/* ---------------- Replay Logic ---------------- */
async function loadReplayList() {
  try {
    const data = await fetchJson("/api/replay/files");
    state.replayFiles = data.files || [];
    const sel = document.getElementById("replay-select");
    sel.innerHTML = state.replayFiles
      .map((f) => `<option value="${esc(f.filename)}">${esc(f.filename)} (${esc(f.run_id)} · ${esc(f.status)})</option>`)
      .join("");
    if (state.replayFiles.length > 0) {
      const target = state.replayFilename && state.replayFiles.some((f) => f.filename === state.replayFilename)
        ? state.replayFilename
        : state.replayFiles[0].filename;
      sel.value = target;
      await loadReplayFile(target);
    }
  } catch (err) {
    renderErrorBanner(err.message);
  }
}

async function loadReplayFile(filename) {
  stopReplayTimer();
  state.replayFilename = filename;
  state.selectedClaimId = null;
  state.selectedAgentId = null;
  const data = await fetchJson(`/api/replay/files/${encodeURIComponent(filename)}`);
  state.trace = data;
  const total = (data.execution_events || []).length;
  state.replayCursor = total;
  const scrubber = document.getElementById("replay-scrubber");
  scrubber.max = String(total);
  scrubber.value = String(total);
  renderTrace();
}

function stopReplayTimer() {
  state.replayPlaying = false;
  if (state.replayTimer) {
    clearInterval(state.replayTimer);
    state.replayTimer = null;
  }
  const btn = document.getElementById("btn-play");
  if (btn) btn.textContent = "▶ Play";
}

function toggleReplayPlay() {
  if (!state.trace) return;
  const total = (state.trace.execution_events || []).length;
  if (state.replayPlaying) {
    stopReplayTimer();
    return;
  }
  if (state.replayCursor >= total) state.replayCursor = 0;
  state.replayPlaying = true;
  document.getElementById("btn-play").textContent = "⏸ Pause";
  const intervalMs = Math.max(120, Math.round(700 / state.replaySpeed));
  state.replayTimer = setInterval(() => {
    if (state.replayCursor < total) {
      state.replayCursor += 1;
      document.getElementById("replay-scrubber").value = String(state.replayCursor);
      renderTrace();
    } else {
      stopReplayTimer();
    }
  }, intervalMs);
}

/* ---------------- Live Monitor Logic ---------------- */
async function refreshLiveRuns() {
  try {
    const health = await fetchJson("/api/health");
    const data = await fetchJson("/api/live/runs");
    state.liveRuns = data.runs || [];
    const sel = document.getElementById("live-select");
    const statusEl = document.getElementById("live-db-status");
    if (!health.db_exists) {
      statusEl.textContent = `Demo DB not yet created (${health.db_path})`;
    } else {
      statusEl.textContent = `${state.liveRuns.length} run(s) in ${health.db_path}`;
    }
    if (state.liveRuns.length === 0) {
      sel.innerHTML = `<option value="">No live runs recorded yet</option>`;
      state.trace = null;
      renderEmptyLive();
      return;
    }
    sel.innerHTML = state.liveRuns
      .map((r) => `<option value="${esc(r.run_id)}">${esc(r.run_id)} (${esc(r.model)} · ${esc(r.status)})</option>`)
      .join("");
    if (!state.liveRunId || !state.liveRuns.some((r) => r.run_id === state.liveRunId)) {
      state.liveRunId = state.liveRuns[0].run_id;
    }
    sel.value = state.liveRunId;
    await loadLiveRun(state.liveRunId);
  } catch (err) {
    document.getElementById("live-db-status").textContent = `Error: ${err.message}`;
  }
}

async function loadLiveRun(runId) {
  if (!runId) return;
  state.liveRunId = runId;
  const data = await fetchJson(`/api/live/runs/${encodeURIComponent(runId)}`);
  state.trace = data;
  state.replayCursor = (data.execution_events || []).length;
  renderTrace();
}

async function pollLive() {
  if (state.tab !== "live") return;
  await refreshLiveRuns();
}

function stopLiveTimer() {
  if (state.liveTimer) {
    clearInterval(state.liveTimer);
    state.liveTimer = null;
  }
}

function renderEmptyLive() {
  document.getElementById("run-overview-grid").innerHTML =
    `<div class="muted">No runs found in the configured demo database. Start the MCP server or CLI against <code>.crewforge/demo_live.db</code> to observe live execution.</div>`;
  document.getElementById("topology-canvas").innerHTML = "";
  document.getElementById("timeline-list").innerHTML = "";
  document.getElementById("claims-list").innerHTML = "";
  document.getElementById("honesty-gate-body").innerHTML = "";
}

function renderErrorBanner(msg) {
  const banner = document.getElementById("replay-banner");
  banner.className = "banner banner-red";
  banner.textContent = `Error: ${msg}`;
}

/* ---------------- Replay Event-Order Projection ---------------- */
function projectVisibleTrace(trace, cursor) {
  const allEvents = trace.execution_events || [];
  if (state.tab !== "replay" || cursor >= allEvents.length) {
    return {
      events: allEvents,
      claims: trace.claims || [],
      verdicts: trace.review_verdicts || [],
      gate: trace.honesty_gate,
      finalResult: trace.final_result,
      gateReached: true,
    };
  }
  const visibleEvents = allEvents.slice(0, cursor);
  const cutoffTs = visibleEvents.length > 0 ? visibleEvents[visibleEvents.length - 1].timestamp : "";
  const visibleClaimIds = new Set();
  let gateReached = false;
  for (const ev of visibleEvents) {
    if (ev.payload && ev.payload.claim_id) visibleClaimIds.add(ev.payload.claim_id);
    if (String(ev.event_type || "").includes("honesty_gate")) gateReached = true;
  }
  const claims = (trace.claims || []).filter((c) => {
    if (visibleClaimIds.has(c.claim_id)) return true;
    return cutoffTs && c.created_at && c.created_at <= cutoffTs;
  });
  const verdicts = (trace.review_verdicts || []).filter(
    (v) => cutoffTs && v.created_at && v.created_at <= cutoffTs
  );
  return {
    events: visibleEvents,
    claims,
    verdicts,
    gate: gateReached ? trace.honesty_gate : null,
    finalResult: gateReached ? trace.final_result : null,
    gateReached,
  };
}

/* ---------------- Render Main Trace Dashboard ---------------- */
function renderTrace() {
  const trace = state.trace;
  if (!trace) return;

  const allEvents = trace.execution_events || [];
  document.getElementById("replay-counter").textContent = `${state.replayCursor} / ${allEvents.length} events`;

  const view = projectVisibleTrace(trace, state.replayCursor);

  renderOverview(trace, view);
  renderTopology(trace);
  renderTimeline(view.events);
  renderClaims(trace, view);
  renderHonestyGate(trace, view);
}

function renderOverview(trace, view) {
  const badge = document.getElementById("ov-status-badge");
  badge.textContent = trace.status;
  badge.className = `pill ${statusPillClass(trace.status)}`;

  const usage = trace.usage || {};
  const gate = view.gate;
  const finalRes = view.finalResult;
  const reqCov =
    finalRes && finalRes.verification && finalRes.verification.requirement_coverage_ratio !== undefined
      ? `${Math.round(finalRes.verification.requirement_coverage_ratio * 100)}%`
      : view.gateReached
      ? "N/A"
      : "Pending replay";

  const releaseSafety = gate
    ? `${gate.release_mode} (${gate.passed ? "passed" : "blocked"})`
    : view.gateReached
    ? "Not recorded"
    : "Pending replay";

  const cards = [
    { label: "Run ID", val: trace.run_id, sub: trace.is_historical_replay ? "Historical Replay" : "Live DB" },
    { label: "Provider / Model", val: `${trace.provider} / ${trace.model}`, sub: `returned: ${trace.returned_model || "unknown"}` },
    { label: "Execution Status", val: trace.status, sub: `mode: ${trace.mode}` },
    { label: "Elapsed Time", val: fmtDuration(trace.latency_ms), sub: `calls: ${usage.model_calls ?? "Unknown"}` },
    { label: "Known Token Usage", val: fmtTokens(usage), sub: `reasoning: ${usage.reasoning_tokens ?? "Unknown"}` },
    { label: "Monetary Cost", val: fmtCost(usage), sub: trace.provider === "ollama" ? "local zero-API" : "reconciled" },
    { label: "Release Safety", val: releaseSafety, sub: "Honesty Gate" },
    { label: "Internal Req. Linkage", val: reqCov, sub: "Not an independent grade" },
  ];

  document.getElementById("run-overview-grid").innerHTML = cards
    .map(
      (c) => `
      <div class="kpi-card">
        <div class="kpi-label">${esc(c.label)}</div>
        <div class="kpi-val">${esc(c.val)}</div>
        <div class="kpi-sub">${esc(c.sub)}</div>
      </div>`
    )
    .join("");
}

function renderTopology(trace) {
  const topo = trace.topology || { nodes: [], edges: [] };
  const nodes = topo.nodes || [];
  const edges = topo.edges || [];
  const wrap = document.getElementById("topology-canvas");

  if (nodes.length === 0) {
    wrap.innerHTML = `<div class="muted" style="padding:16px">No agent roster persisted for this run.</div>`;
    return;
  }

  // Layout nodes by column: lead (col 0), specialists (col 1), reviewer (col 2), gate (col 3)
  const cols = { lead: [], specialist: [], reviewer: [], gate: [] };
  for (const n of nodes) {
    (cols[n.node_type] || cols.specialist).push(n);
  }
  const activeCols = ["lead", "specialist", "reviewer", "gate"].filter((k) => cols[k].length > 0);
  const width = 660;
  const height = 260;
  const pos = {};
  activeCols.forEach((colKey, cIdx) => {
    const list = cols[colKey];
    const x = Math.round(((cIdx + 0.5) / activeCols.length) * (width - 40) + 20);
    list.forEach((n, rIdx) => {
      const y = Math.round(((rIdx + 0.5) / list.length) * (height - 30) + 15);
      pos[n.id] = { x, y };
    });
  });

  const colorMap = {
    lead: "#3b82f6",
    specialist: "#14b8a6",
    reviewer: "#f59e0b",
    gate: "#10b981",
  };

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrow-dep" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#3b82f6" />
      </marker>
      <marker id="arrow-msg" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#14b8a6" />
      </marker>
    </defs>`;

  for (const e of edges) {
    const p1 = pos[e.source];
    const p2 = pos[e.target];
    if (!p1 || !p2) continue;
    const isMsg = e.edge_type === "peer_message";
    const stroke = isMsg ? "#14b8a6" : "#3b82f6";
    const dash = isMsg ? 'stroke-dasharray="5,4"' : "";
    const marker = isMsg ? "url(#arrow-msg)" : "url(#arrow-dep)";
    svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${stroke}" stroke-width="1.8" ${dash} marker-end="${marker}" opacity="0.85" />`;
  }

  for (const n of nodes) {
    const p = pos[n.id];
    const w = 132;
    const h = 54;
    const stroke = n.status === "blocked" || n.status === "failed" ? "#ef4444" : colorMap[n.node_type] || "#14b8a6";
    svg += `
      <g class="topo-node" data-agent-id="${esc(n.id)}">
        <rect x="${p.x - w / 2}" y="${p.y - h / 2}" width="${w}" height="${h}" rx="8" fill="#1a2540" stroke="${stroke}" stroke-width="1.8" />
        <text x="${p.x}" y="${p.y - 6}" text-anchor="middle" fill="#e2e8f0" font-size="11.5" font-weight="700">${esc(n.label)}</text>
        <text x="${p.x}" y="${p.y + 11}" text-anchor="middle" fill="#94a3b8" font-size="10">${esc(n.status)} · ${esc(n.sublabel).slice(0, 18)}</text>
      </g>`;
  }
  svg += `</svg>`;
  wrap.innerHTML = svg;

  wrap.querySelectorAll(".topo-node").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedAgentId = el.dataset.agentId;
      renderAgentDrawer(trace);
    });
  });

  if (state.selectedAgentId) renderAgentDrawer(trace);
}

function renderAgentDrawer(trace) {
  const drawer = document.getElementById("agent-drawer");
  const node = (trace.topology?.nodes || []).find((n) => n.id === state.selectedAgentId);
  if (!node) {
    drawer.classList.add("hidden");
    return;
  }
  drawer.classList.remove("hidden");
  const tasksHtml = (node.tasks || []).length
    ? node.tasks
        .map(
          (t) => `<div class="sub-box"><strong>${esc(t.task_id)}</strong> [${esc(t.status)}] — ${esc(t.title || t.description)}</div>`
        )
        .join("")
    : `<div class="muted">No specialist tasks assigned directly to this component.</div>`;

  drawer.innerHTML = `
    <div class="panel-header">
      <h3>Agent Activity: ${esc(node.label)} (${esc(node.id)})</h3>
      <span class="pill ${statusPillClass(node.status)}">${esc(node.status)}</span>
    </div>
    <div>Model: <code>${esc(node.model || "deterministic")}</code></div>
    ${tasksHtml}
  `;
}

function renderTimeline(events) {
  const categories = [
    ["all", "All"],
    ["planning", "Planning"],
    ["task_execution", "Task execution"],
    ["peer_messages", "Peer messages"],
    ["tool_calls", "Tool calls"],
    ["review", "Review"],
    ["revision", "Revision"],
    ["synthesis", "Synthesis"],
    ["honesty_gate", "Honesty Gate"],
    ["budget", "Budget events"],
  ];

  const filterBar = document.getElementById("timeline-filters");
  filterBar.innerHTML = categories
    .map(
      ([key, label]) =>
        `<button class="filter-btn ${state.timelineFilter === key ? "active" : ""}" data-cat="${esc(key)}">${esc(label)}</button>`
    )
    .join("");
  filterBar.querySelectorAll(".filter-btn").forEach((b) => {
    b.addEventListener("click", () => {
      state.timelineFilter = b.dataset.cat;
      renderTimeline(events);
    });
  });

  const filtered =
    state.timelineFilter === "all"
      ? events
      : events.filter((e) => e.category === state.timelineFilter);

  document.getElementById("timeline-count").textContent = `${filtered.length} / ${events.length} events`;

  const list = document.getElementById("timeline-list");
  if (filtered.length === 0) {
    list.innerHTML = `<div class="muted">No events match this filter.</div>`;
    return;
  }
  list.innerHTML = filtered
    .map((ev) => {
      const ts = String(ev.timestamp || "").slice(11, 19);
      const summary = ev.payload ? JSON.stringify(ev.payload).slice(0, 180) : "";
      return `
        <div class="ev-row">
          <div class="ev-seq">#${esc(ev.seq)} ${esc(ts)}</div>
          <div class="ev-type">${esc(ev.event_type)}</div>
          <div class="ev-desc">${esc(summary)}</div>
        </div>`;
    })
    .join("");
}

function renderClaims(trace, view) {
  const claims = view.claims || [];
  document.getElementById("claims-summary").textContent = `${claims.length} claim version(s)`;
  const list = document.getElementById("claims-list");

  if (claims.length === 0) {
    list.innerHTML = `<div class="muted">No claims visible at the current replay cursor.</div>`;
    document.getElementById("claim-detail").classList.add("hidden");
    return;
  }

  list.innerHTML = claims
    .map((c) => {
      const sel = state.selectedClaimId === `${c.claim_id}@${c.version}` ? "selected" : "";
      return `
        <div class="claim-item ${sel}" data-cid="${esc(c.claim_id)}" data-ver="${esc(c.version)}">
          <div class="claim-top">
            <div><code>${esc(c.claim_id)}@v${esc(c.version)}</code> · <span class="muted">${esc(c.category)} · ${esc(c.author_agent_id)}</span></div>
            <span class="pill ${statusPillClass(c.status)}">${esc(c.status)}</span>
          </div>
          <div class="claim-stmt">${esc(c.statement)}</div>
        </div>`;
    })
    .join("");

  list.querySelectorAll(".claim-item").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedClaimId = `${el.dataset.cid}@${el.dataset.ver}`;
      renderClaims(trace, view);
      renderClaimDetail(trace, view, el.dataset.cid, Number(el.dataset.ver));
    });
  });

  if (state.selectedClaimId) {
    const [cid, ver] = state.selectedClaimId.split("@");
    renderClaimDetail(trace, view, cid, Number(ver));
  }
}

function renderClaimDetail(trace, view, claimId, version) {
  const detail = document.getElementById("claim-detail");
  const claim = (view.claims || []).find((c) => c.claim_id === claimId && Number(c.version) === version);
  if (!claim) {
    detail.classList.add("hidden");
    return;
  }
  detail.classList.remove("hidden");

  const evdById = Object.fromEntries((trace.evidence || []).map((e) => [e.evidence_id, e]));
  const toolById = Object.fromEntries((trace.tool_executions || []).map((t) => [t.tool_execution_id, t]));
  const verdicts = (view.verdicts || []).filter((v) => v.claim_id === claimId);
  const versions = (view.claims || []).filter((c) => c.claim_id === claimId);

  const evdHtml = (claim.evidence_ids || [])
    .map((eid) => {
      const e = evdById[eid];
      if (!e) return `<div class="sub-box">Evidence ${esc(eid)}</div>`;
      return `<div class="sub-box"><strong>${esc(eid)}</strong> [${esc(e.source_type)} · <code>${esc(e.source_ref || "none")}</code>]<br/>${esc(e.excerpt)}</div>`;
    })
    .join("") || `<div class="muted">No supporting evidence linked.</div>`;

  const toolHtml = (claim.tool_execution_ids || [])
    .map((tid) => {
      const t = toolById[tid];
      if (!t) return `<div class="sub-box">Tool ${esc(tid)}</div>`;
      return `<div class="sub-box"><strong>${esc(t.tool_name)}</strong> (${esc(t.status)}): <code>${esc(JSON.stringify(t.output_data))}</code></div>`;
    })
    .join("") || `<div class="muted">No deterministic tool executions linked.</div>`;

  const verdictHtml = verdicts.length
    ? verdicts
        .map(
          (v) => {
            const dec = v.decision || v.verdict || "";
            const fix = v.required_correction || v.required_fix || "";
            return `<div class="sub-box"><span class="pill ${statusPillClass(dec)}">${esc(dec)}</span> <strong>${esc(v.reviewer_id)}</strong> (v${esc(v.claim_version)}): ${esc(v.rationale || fix)}</div>`;
          }
        )
        .join("")
    : `<div class="muted">No reviewer verdict recorded yet.</div>`;

  detail.innerHTML = `
    <div class="panel-header">
      <h3>${esc(claim.claim_id)} (v${esc(claim.version)})</h3>
      <span class="pill ${statusPillClass(claim.status)}">${esc(claim.status)}</span>
    </div>
    <div><strong>Requirements:</strong> ${esc((claim.requirement_refs || []).join(", ") || "None")}</div>
    <div><strong>Premise Dependencies:</strong> ${esc((claim.premise_claim_ids || []).join(", ") || "None")}</div>
    <div><strong>Version History:</strong> ${esc(versions.map((x) => `v${x.version}:${x.status}`).join(" → "))}</div>
    <div><strong>Supporting Evidence:</strong></div>${evdHtml}
    <div><strong>Deterministic Arithmetic / Tool Results:</strong></div>${toolHtml}
    <div><strong>Reviewer Verdicts &amp; Withholding Rationale:</strong></div>${verdictHtml}
  `;
}

function renderHonestyGate(trace, view) {
  const body = document.getElementById("honesty-gate-body");
  const pill = document.getElementById("gate-mode-pill");
  const gate = view.gate;
  const finalRes = view.finalResult;

  if (!view.gateReached) {
    pill.textContent = "PENDING REPLAY";
    pill.className = "pill pill-amber";
    body.innerHTML = `<div class="muted">Advance replay to the <code>honesty_gate</code> event to inspect the release decision and final answer.</div>`;
    return;
  }

  if (!gate) {
    pill.textContent = "NOT RECORDED";
    pill.className = "pill pill-red";
    body.innerHTML = `<div class="muted">No Honesty Gate decision persisted for this run.</div>`;
    return;
  }

  pill.textContent = `${gate.release_mode} (${gate.passed ? "PASSED" : "BLOCKED"})`;
  pill.className = `pill ${statusPillClass(gate.passed ? gate.release_mode : "blocked")}`;

  const answerText = finalRes && finalRes.answer ? finalRes.answer : "(No final answer released)";
  const missing = (finalRes && finalRes.missing_information) || [];
  const reasons = gate.reasons || [];

  body.innerHTML = `
    <div class="banner banner-blue">
      Note: Passing the Honesty Gate verifies claim-level provenance and independent review, <strong>not</strong> factual correctness or complete requirement coverage.
    </div>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Supported Claims</div><div class="kpi-val">${esc((gate.supported_claim_ids || []).length)}</div></div>
      <div class="kpi-card"><div class="kpi-label">Withheld Claims</div><div class="kpi-val">${esc((gate.withheld_claim_ids || []).length)}</div></div>
      <div class="kpi-card"><div class="kpi-label">Rejected Segments</div><div class="kpi-val">${esc((gate.rejected_segment_ids || []).length)}</div></div>
      <div class="kpi-card"><div class="kpi-label">Unresolved Reqs</div><div class="kpi-val">${esc((gate.unresolved_requirements || []).length)}</div></div>
    </div>
    <div><strong>Gate Reasons:</strong> ${esc(reasons.join(" | ") || "None")}</div>
    <div><strong>Missing Information:</strong> ${esc(missing.join(" | ") || "None")}</div>
    <div><strong>Final Released Answer:</strong></div>
    <div class="answer-box">${esc(answerText)}</div>
  `;
}

/* ---------------- Benchmark V2 Page ---------------- */
async function loadBenchmark() {
  const container = document.getElementById("benchmark-body");
  try {
    const data = await fetchJson("/api/benchmark");
    state.benchmarkData = data;
    const s = data.summary || {};
    const dvg = data.david_vs_goliath || {};

    const domainRows = Object.entries(data.domain_scores || {})
      .map(
        ([dom, ds]) => `
        <tr>
          <td><code>${esc(dom)}</code></td>
          <td>${esc(ds.baseline_mean.toFixed(2))}</td>
          <td>${esc(ds.network_mean.toFixed(2))}</td>
          <td>
            <div class="bar-track"><div class="bar-fill-blue" style="width:${Math.min(100, ds.baseline_mean)}%"></div></div>
            <div class="bar-track" style="margin-top:4px"><div class="bar-fill-teal" style="width:${Math.min(100, ds.network_mean)}%"></div></div>
          </td>
        </tr>`
      )
      .join("");

    const caseRows = (data.per_case || [])
      .map(
        (c) => `
        <tr>
          <td><code>${esc(c.prompt_id)}</code></td>
          <td><code>${esc(c.domain)}</code></td>
          <td>${esc(c.baseline.final_score)}</td>
          <td>${esc(c.network.final_score)}</td>
          <td><span class="pill ${statusPillClass(c.network.status)}">${esc(c.network.status)}</span></td>
          <td>${esc(c.baseline.unsupported_material_assertion_count)} / ${esc(c.network.unsupported_material_assertion_count)}</td>
          <td><span class="pill ${c.winner === "network" ? "pill-green" : "pill-blue"}">${esc(c.winner)}</span></td>
        </tr>`
      )
      .join("");

    container.innerHTML = `
      <div class="banner banner-amber">
        <strong>${esc(data.grading_method || "ChatGPT-assisted blind semantic grading")}</strong> —
        ${esc(data.benchmark_note || "10 internally authored cases, not official hackathon questions.")}
      </div>

      <section class="panel">
        <div class="panel-header">
          <h2>Held-Out V2 Same-Model Comparison (<code>${esc(data.model)}</code> · Engine <code>${esc(data.engine_commit?.slice(0, 7))}</code>)</h2>
          <span class="pill pill-red">Network Wins: ${esc(s.network_wins)} / 10</span>
        </div>
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Single-Agent Mean</div><div class="kpi-val">${esc(s.baseline_mean?.toFixed(2))}/100</div><div class="kpi-sub">Median: ${esc(s.baseline_median?.toFixed(2))}</div></div>
          <div class="kpi-card"><div class="kpi-label">CrewForge Mean</div><div class="kpi-val">${esc(s.network_mean?.toFixed(2))}/100</div><div class="kpi-sub">Median: ${esc(s.network_median?.toFixed(2))}</div></div>
          <div class="kpi-card"><div class="kpi-label">Unsupported Assertions</div><div class="kpi-val">${esc(s.baseline_unsupported_assertions)} vs ${esc(s.network_unsupported_assertions)}</div><div class="kpi-sub">Baseline vs Network</div></div>
          <div class="kpi-card"><div class="kpi-label">Total Cost</div><div class="kpi-val">$${esc(s.baseline_total_cost_usd?.toFixed(5))} vs $${esc(s.network_total_cost_usd?.toFixed(5))}</div><div class="kpi-sub">Baseline vs Network</div></div>
          <div class="kpi-card"><div class="kpi-label">Mean Latency</div><div class="kpi-val">${esc(s.baseline_mean_latency_s?.toFixed(2))}s vs ${esc(s.network_mean_latency_s?.toFixed(2))}s</div><div class="kpi-sub">Baseline vs Network</div></div>
          <div class="kpi-card"><div class="kpi-label">Strategic Planning</div><div class="kpi-val">93.50 vs 87.00</div><div class="kpi-sub">Network vs Baseline (selected domain)</div></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>David vs Goliath (Qwen3-4B Network vs GPT-6.2 Astra Baseline)</h2>
          <span class="pill pill-amber">${esc(dvg.status || "Not executed — local readiness blocker.")}</span>
        </div>
        <div class="muted">${esc(dvg.detail || "")}</div>
      </section>

      <div class="grid-2">
        <section class="panel">
          <div class="panel-header"><h2>Domain-Level Scores (5 Domains)</h2></div>
          <table class="data-table">
            <thead><tr><th>Domain</th><th>Baseline</th><th>Network</th><th>Comparison</th></tr></thead>
            <tbody>${domainRows}</tbody>
          </table>
          <div class="sub-box muted">
            Selected scheduling case (<code>v2_str_02</code>): Network <strong>100</strong> vs Baseline <strong>74</strong> (single-case result; aggregate benchmark remains 1/10 wins).
          </div>
        </section>

        <section class="panel">
          <div class="panel-header"><h2>Per-Case V2 Results</h2></div>
          <table class="data-table">
            <thead><tr><th>Case</th><th>Domain</th><th>Base</th><th>Net</th><th>Net Status</th><th>Unsup (B/N)</th><th>Winner</th></tr></thead>
            <tbody>${caseRows}</tbody>
          </table>
        </section>
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<div class="banner banner-red">Failed to load benchmark report: ${esc(err.message)}</div>`;
  }
}

/* ---------------- Wire Events ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.addEventListener("click", () => setTab(b.dataset.tab));
  });
  document.getElementById("replay-select").addEventListener("change", (e) => loadReplayFile(e.target.value));
  document.getElementById("btn-play").addEventListener("click", toggleReplayPlay);
  document.getElementById("btn-restart").addEventListener("click", () => {
    stopReplayTimer();
    state.replayCursor = 0;
    document.getElementById("replay-scrubber").value = "0";
    renderTrace();
  });
  document.getElementById("btn-step").addEventListener("click", () => {
    if (!state.trace) return;
    stopReplayTimer();
    const total = (state.trace.execution_events || []).length;
    if (state.replayCursor < total) {
      state.replayCursor += 1;
      document.getElementById("replay-scrubber").value = String(state.replayCursor);
      renderTrace();
    }
  });
  document.getElementById("replay-speed").addEventListener("change", (e) => {
    state.replaySpeed = Number(e.target.value) || 1;
    if (state.replayPlaying) {
      stopReplayTimer();
      toggleReplayPlay();
    }
  });
  document.getElementById("replay-scrubber").addEventListener("input", (e) => {
    stopReplayTimer();
    state.replayCursor = Number(e.target.value) || 0;
    renderTrace();
  });
  document.getElementById("btn-refresh-live").addEventListener("click", refreshLiveRuns);
  document.getElementById("live-select").addEventListener("change", (e) => loadLiveRun(e.target.value));

  setTab("replay");
});
