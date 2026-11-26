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
  followNewest: true,
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
  if (s === "completed" || s === "supported" || s === "approve" || s === "full" || s === "passed" || s === "answered") {
    return "pill-green";
  }
  if (s === "partial" || s === "partially_answered" || s === "unresolved" || s === "needs_revision" || s === "revise" || s === "stale") {
    return "pill-amber";
  }
  if (s === "failed" || s === "rejected" || s === "reject" || s === "blocked") {
    return "pill-red";
  }
  if (s === "running" || s === "in_progress") {
    return "pill-blue";
  }
  return "pill-teal";
}

function eventTime(ev) {
  return ev?.created_at || ev?.timestamp || "";
}

function fmtTokens(usage) {
  if (!usage || usage.usage_reported === false) return "Unknown (not reported)";
  const inp = usage.input_tokens;
  const out = usage.output_tokens;
  if (inp === null || inp === undefined || out === null || out === undefined) return "Unknown";
  return `${Number(inp).toLocaleString()} in / ${Number(out).toLocaleString()} out`;
}

function fmtCost(usage, provider) {
  if (!usage) return "Unknown";
  const act = usage.actual_cost_usd;
  if (act === null || act === undefined) {
    return provider === "ollama" ? "$0.000000 (local)" : "Unknown";
  }
  return `$${Number(act).toFixed(6)}`;
}

function fmtDuration(ms) {
  if (ms === null || ms === undefined) return "Unknown";
  const n = Number(ms);
  if (Number.isNaN(n)) return "Unknown";
  return `${(n / 1000).toFixed(2)} s`;
}

function setConnectionStatus(ok, label) {
  const el = document.getElementById("connection-pill");
  if (!el) return;
  el.textContent = label || (ok ? "Connected" : "Disconnected");
  el.className = `pill ${ok ? "pill-green" : "pill-red"}`;
}

async function fetchJson(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  setConnectionStatus(true, "Connected");
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
      "<strong>HISTORICAL REPLAY</strong> — Reconstructing persisted trace in recorded event order. Future claims, verdicts, usage, and final answer remain hidden until their corresponding events.";
    modeInd.textContent = "HISTORICAL REPLAY";
    modeInd.className = "pill pill-amber";
    loadReplayList();
  } else {
    replayCtrls.classList.add("hidden");
    liveCtrls.classList.remove("hidden");
    banner.className = "banner banner-blue";
    banner.innerHTML =
      "<strong>LIVE MONITOR</strong> — Polling read-only configured demo SQLite database (<code>mode=ro</code>) for active or newly created runs.";
    modeInd.textContent = "LIVE MONITOR";
    modeInd.className = "pill pill-blue";
    refreshLiveRuns();
    state.liveTimer = setInterval(pollLive, 2000);
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
      const target =
        state.replayFilename && state.replayFiles.some((f) => f.filename === state.replayFilename)
          ? state.replayFilename
          : state.replayFiles[0].filename;
      sel.value = target;
      await loadReplayFile(target);
    }
  } catch (err) {
    setConnectionStatus(false, "Error");
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
      statusEl.textContent = `Demo DB awaiting first run (${health.db_path})`;
    } else {
      statusEl.textContent = `${state.liveRuns.length} run(s) in ${health.db_path} (read-only)`;
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
    if (state.followNewest || !state.liveRunId || !state.liveRuns.some((r) => r.run_id === state.liveRunId)) {
      state.liveRunId = state.liveRuns[0].run_id;
    }
    sel.value = state.liveRunId;
    await loadLiveRun(state.liveRunId);
  } catch (err) {
    setConnectionStatus(false, "Polling error");
    document.getElementById("live-db-status").textContent = `Error: ${err.message}`;
  }
}

async function loadLiveRun(runId) {
  if (!runId) return;
  state.liveRunId = runId;
  const data = await fetchJson(`/api/live/runs/${encodeURIComponent(runId)}`);
  const prev = state.trace;
  const sameRun = prev && !prev.is_historical_replay && prev.run_id === data.run_id;
  const sameLen = sameRun && (prev.execution_events || []).length === (data.execution_events || []).length;
  const sameStatus = sameRun && prev.status === data.status;
  if (sameLen && sameStatus) return;
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
  document.getElementById("ov-status-badge").textContent = "AWAITING RUN";
  document.getElementById("ov-status-badge").className = "pill pill-teal";
  document.getElementById("run-overview-grid").innerHTML =
    `<div class="muted">No runs found in the configured demo database. Start <code>crewforge-mcp</code> or <code>crewforge solve</code> against <code>.crewforge/demo_live.db</code> to observe live execution.</div>`;
  document.getElementById("topology-canvas").innerHTML = "";
  document.getElementById("agent-drawer").classList.add("hidden");
  document.getElementById("timeline-list").innerHTML = "";
  document.getElementById("claims-list").innerHTML = "";
  document.getElementById("claim-detail").classList.add("hidden");
  document.getElementById("honesty-gate-body").innerHTML = "";
}

function renderErrorBanner(msg) {
  const banner = document.getElementById("replay-banner");
  banner.className = "banner banner-red";
  banner.textContent = `Error: ${msg}`;
}

/* ---------------- Strict Event-Order Replay Projection ---------------- */
function projectVisibleTrace(trace, cursor) {
  const allEvents = trace.execution_events || [];
  const isFullReplay = state.tab !== "replay" || cursor >= allEvents.length;

  if (isFullReplay) {
    return {
      events: allEvents,
      planRevealed: Boolean(trace.execution_plan || (trace.roster || []).length > 0),
      roster: trace.roster || [],
      tasks: trace.tasks || [],
      messages: trace.peer_messages || [],
      evidence: trace.evidence || [],
      tools: trace.tool_executions || [],
      claims: trace.claims || [],
      verdicts: trace.review_verdicts || [],
      gate: trace.honesty_gate,
      finalResult: trace.final_result,
      gateReached: Boolean(trace.honesty_gate || trace.final_result),
      runCompleted: ["completed", "partial", "failed"].includes(String(trace.status || "").toLowerCase()),
      visibleStatus: trace.status,
      visibleUsage: trace.usage,
      visibleLatencyMs: trace.latency_ms,
      transitionsAvailable: (trace.task_transitions || []).length > 0,
    };
  }

  const visibleEvents = allEvents.slice(0, cursor);
  const cutoffTs = visibleEvents.length > 0 ? eventTime(visibleEvents[visibleEvents.length - 1]) : "";
  const evTypes = new Set(visibleEvents.map((e) => String(e.event_type || "")));

  const planRevealed = evTypes.has("plan_created");
  const gateReached = evTypes.has("honesty_gate_evaluated") || evTypes.has("run_completed") || evTypes.has("run_failed");
  const runCompleted = evTypes.has("run_completed") || evTypes.has("run_failed");

  const roster = planRevealed
    ? trace.roster || []
    : (trace.roster || []).filter((a) => String(a.role || "").toLowerCase() === "lead");

  // Reconstruct task statuses from task_transitions <= cutoffTs
  const transitions = (trace.task_transitions || []).filter(
    (tr) => cutoffTs && tr.created_at && tr.created_at <= cutoffTs
  );
  const taskStatusById = {};
  for (const tr of transitions) {
    if (tr.task_id && tr.to_status) taskStatusById[tr.task_id] = tr.to_status;
  }
  const tasks = planRevealed
    ? (trace.tasks || []).map((t) => ({
        ...t,
        status: taskStatusById[t.task_id] || (transitions.length > 0 ? "pending" : "unavailable"),
      }))
    : [];

  const messages = (trace.peer_messages || []).filter(
    (m) => cutoffTs && m.created_at && m.created_at <= cutoffTs
  );
  const evidence = (trace.evidence || []).filter(
    (e) => cutoffTs && e.created_at && e.created_at <= cutoffTs
  );
  const tools = (trace.tool_executions || []).filter(
    (t) => cutoffTs && t.created_at && t.created_at <= cutoffTs
  );
  const claims = (trace.claims || []).filter(
    (c) => cutoffTs && c.created_at && c.created_at <= cutoffTs
  );
  const verdicts = (trace.review_verdicts || []).filter(
    (v) => cutoffTs && v.created_at && v.created_at <= cutoffTs
  );

  // Incremental usage & elapsed time from visible events
  let calls = 0;
  let inTok = 0;
  let outTok = 0;
  let reasTok = 0;
  let cost = 0;
  let anyUsage = false;
  for (const ev of visibleEvents) {
    if (ev.event_type === "model_call_completed" && ev.payload) {
      anyUsage = true;
      calls += 1;
      inTok += Number(ev.payload.input_tokens || 0);
      outTok += Number(ev.payload.output_tokens || 0);
      reasTok += Number(ev.payload.reasoning_tokens || 0);
      cost += Number(ev.payload.actual_cost_usd || 0);
    }
  }
  const visibleUsage = runCompleted
    ? trace.usage
    : anyUsage
    ? {
        input_tokens: inTok,
        output_tokens: outTok,
        reasoning_tokens: reasTok,
        actual_cost_usd: cost,
        model_calls: calls,
        usage_reported: true,
      }
    : { usage_reported: false, model_calls: calls };

  let visibleLatencyMs = null;
  if (runCompleted) {
    visibleLatencyMs = trace.latency_ms;
  } else if (visibleEvents.length > 0 && trace.created_at && cutoffTs) {
    const startMs = Date.parse(trace.created_at);
    const endMs = Date.parse(cutoffTs);
    if (!Number.isNaN(startMs) && !Number.isNaN(endMs) && endMs >= startMs) {
      visibleLatencyMs = endMs - startMs;
    }
  }

  let visibleStatus = "pending";
  if (evTypes.has("run_failed")) visibleStatus = "failed";
  else if (evTypes.has("run_completed")) visibleStatus = trace.status;
  else if (visibleEvents.length > 0) visibleStatus = "running";

  return {
    events: visibleEvents,
    planRevealed,
    roster,
    tasks,
    messages,
    evidence,
    tools,
    claims,
    verdicts,
    gate: gateReached ? trace.honesty_gate : null,
    finalResult: runCompleted ? trace.final_result : null,
    gateReached,
    runCompleted,
    visibleStatus,
    visibleUsage,
    visibleLatencyMs,
    transitionsAvailable: (trace.task_transitions || []).length > 0,
  };
}

/* ---------------- Render Main Trace Dashboard ---------------- */
function renderTrace() {
  const trace = state.trace;
  if (!trace) return;

  const allEvents = trace.execution_events || [];
  document.getElementById("replay-counter").textContent = `${state.replayCursor} / ${allEvents.length} events`;

  const view = projectVisibleTrace(trace, state.replayCursor);

  // Clear stale selections if backward-scrubbed
  if (state.selectedAgentId) {
    const validIds = new Set((view.roster || []).map((a) => a.agent_id));
    if (view.verdicts.length > 0) view.verdicts.forEach((v) => validIds.add(v.reviewer_id));
    if (view.gate) validIds.add("honesty_gate");
    if (!validIds.has(state.selectedAgentId)) state.selectedAgentId = null;
  }
  if (state.selectedClaimId) {
    const validClaims = new Set((view.claims || []).map((c) => `${c.claim_id}@${c.version}`));
    if (!validClaims.has(state.selectedClaimId)) state.selectedClaimId = null;
  }

  renderOverview(trace, view);
  renderTopology(trace, view);
  renderTimeline(view.events);
  renderClaims(trace, view);
  renderHonestyGate(trace, view);
}

function renderOverview(trace, view) {
  const badge = document.getElementById("ov-status-badge");
  badge.textContent = view.visibleStatus;
  badge.className = `pill ${statusPillClass(view.visibleStatus)}`;

  const usage = view.visibleUsage || {};
  const gate = view.gate;
  const verSummary = gate?.verification_summary || view.finalResult?.verification || null;
  const reqCovVal = verSummary?.requirement_coverage;
  const reqCov =
    reqCovVal !== null && reqCovVal !== undefined
      ? `${(Number(reqCovVal) * 100).toFixed(1)}%`
      : view.gateReached
      ? "Unknown"
      : "Pending replay";

  let releaseSafety = "Pending replay";
  if (gate) {
    const safeTag = verSummary?.content_released_safely === true ? "safe" : gate.passed ? "passed" : "blocked";
    releaseSafety = `${gate.release_mode} (${safeTag})`;
  } else if (view.gateReached) {
    releaseSafety = "Not recorded";
  }

  const cards = [
    { label: "Run ID", val: trace.run_id, sub: trace.is_historical_replay ? "Historical Replay" : "Live Read-Only DB" },
    { label: "Provider / Model", val: `${trace.provider} / ${trace.model}`, sub: `returned: ${trace.returned_model || "unknown"}` },
    { label: "Execution Status", val: view.visibleStatus, sub: `mode: ${trace.mode}` },
    { label: "Elapsed Time", val: fmtDuration(view.visibleLatencyMs), sub: `calls: ${usage.model_calls ?? "Unknown"}` },
    { label: "Known Token Usage", val: fmtTokens(usage), sub: `reasoning: ${usage.reasoning_tokens ?? "Unknown"}` },
    { label: "Monetary Cost", val: fmtCost(usage, trace.provider), sub: trace.provider === "ollama" ? "local zero-API" : "reconciled" },
    { label: "Release Safety", val: releaseSafety, sub: gate ? `gate passed: ${gate.passed}` : "Honesty Gate" },
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

function buildProjectedTopology(trace, view) {
  const nodes = [];
  const seen = new Set();

  for (const a of view.roster || []) {
    const aid = a.agent_id;
    if (!aid || seen.has(aid)) continue;
    seen.add(aid);
    const role = String(a.role || "specialist").toLowerCase();
    const assigned = (view.tasks || []).filter((t) => t.assigned_agent_id === aid);
    let status = "pending";
    if (assigned.length === 0) {
      status = role === "lead" ? view.visibleStatus : "pending";
    } else {
      const sts = new Set(assigned.map((t) => String(t.status || "pending").toLowerCase()));
      if (sts.has("failed")) status = "failed";
      else if (sts.has("blocked")) status = "blocked";
      else if (sts.has("running") || sts.has("in_progress")) status = "running";
      else if ([...sts].every((s) => s === "completed")) status = "completed";
      else if (sts.has("unavailable")) status = "unavailable";
    }
    nodes.push({
      id: aid,
      label: a.role || aid,
      sublabel: a.specialty || a.model_override || a.model || "",
      node_type: role === "lead" ? "lead" : "specialist",
      status,
      model: a.model_override || a.model || trace.model,
      tasks: assigned,
    });
  }

  if ((view.verdicts || []).length > 0) {
    const rids = [...new Set(view.verdicts.map((v) => v.reviewer_id || "reviewer"))];
    for (const rid of rids) {
      if (seen.has(rid)) continue;
      seen.add(rid);
      const count = view.verdicts.filter((v) => (v.reviewer_id || "reviewer") === rid).length;
      nodes.push({
        id: rid,
        label: rid,
        sublabel: `${count} verdict(s)`,
        node_type: "reviewer",
        status: "completed",
        model: trace.model,
        tasks: [],
      });
    }
  }

  if (view.gate) {
    const rmode = String(view.gate.release_mode || "unknown").toLowerCase();
    nodes.push({
      id: "honesty_gate",
      label: "Honesty Gate",
      sublabel: `mode: ${view.gate.release_mode || "unknown"}`,
      node_type: "gate",
      status: !view.gate.passed || rmode === "blocked" ? "blocked" : rmode === "partial" ? "partial" : "completed",
      model: null,
      tasks: [],
    });
    seen.add("honesty_gate");
  }

  const edges = [];
  const edgeKeys = new Set();
  const leadId = nodes.find((n) => n.node_type === "lead")?.id;
  const taskById = Object.fromEntries((view.tasks || []).map((t) => [t.task_id, t]));

  for (const t of view.tasks || []) {
    const deps = t.depends_on || t.depends_on_task_ids || [];
    if (deps.length === 0 && leadId && t.assigned_agent_id && leadId !== t.assigned_agent_id) {
      const k = `${leadId}->${t.assigned_agent_id}:dep`;
      if (!edgeKeys.has(k)) {
        edgeKeys.add(k);
        edges.push({ source: leadId, target: t.assigned_agent_id, edge_type: "task_dependency" });
      }
    }
    for (const depTid of deps) {
      const srcAgent = taskById[depTid]?.assigned_agent_id;
      if (srcAgent && t.assigned_agent_id && srcAgent !== t.assigned_agent_id) {
        const k = `${srcAgent}->${t.assigned_agent_id}:dep`;
        if (!edgeKeys.has(k)) {
          edgeKeys.add(k);
          edges.push({ source: srcAgent, target: t.assigned_agent_id, edge_type: "task_dependency" });
        }
      }
    }
  }

  for (const m of view.messages || []) {
    const src = m.sender_id || m.sender_agent_id;
    const dst = m.recipient_id || m.recipient_agent_id;
    if (seen.has(src) && seen.has(dst) && src !== dst) {
      const k = `${src}->${dst}:msg`;
      if (!edgeKeys.has(k)) {
        edgeKeys.add(k);
        edges.push({ source: src, target: dst, edge_type: "peer_message" });
      }
    }
  }

  return { nodes, edges };
}

function renderTopology(trace, view) {
  const topo = buildProjectedTopology(trace, view);
  const nodes = topo.nodes;
  const edges = topo.edges;
  const wrap = document.getElementById("topology-canvas");

  if (nodes.length === 0) {
    wrap.innerHTML = `<div class="muted" style="padding:16px">${
      view.planRevealed ? "No agent roster persisted for this run." : "Pending planning event (plan_created)..."
    }</div>`;
    document.getElementById("agent-drawer").classList.add("hidden");
    return;
  }

  const cols = { lead: [], specialist: [], reviewer: [], gate: [] };
  for (const n of nodes) (cols[n.node_type] || cols.specialist).push(n);
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

  const colorMap = { lead: "#3b82f6", specialist: "#14b8a6", reviewer: "#f59e0b", gate: "#10b981" };

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
    svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${isMsg ? "#14b8a6" : "#3b82f6"}" stroke-width="1.8" ${isMsg ? 'stroke-dasharray="5,4"' : ""} marker-end="${isMsg ? "url(#arrow-msg)" : "url(#arrow-dep)"}" opacity="0.85" />`;
  }

  for (const n of nodes) {
    const p = pos[n.id];
    const w = 136;
    const h = 54;
    const stroke =
      n.status === "blocked" || n.status === "failed"
        ? "#ef4444"
        : n.status === "partial"
        ? "#f59e0b"
        : colorMap[n.node_type] || "#14b8a6";
    svg += `
      <g class="topo-node" data-agent-id="${esc(n.id)}">
        <rect x="${p.x - w / 2}" y="${p.y - h / 2}" width="${w}" height="${h}" rx="8" fill="#1a2540" stroke="${stroke}" stroke-width="1.8" />
        <text x="${p.x}" y="${p.y - 6}" text-anchor="middle" fill="#e2e8f0" font-size="11.5" font-weight="700">${esc(String(n.label).slice(0, 20))}</text>
        <text x="${p.x}" y="${p.y + 11}" text-anchor="middle" fill="#94a3b8" font-size="10">${esc(n.status)} · ${esc(String(n.sublabel).slice(0, 16))}</text>
      </g>`;
  }
  svg += `</svg>`;
  wrap.innerHTML = svg;

  wrap.querySelectorAll(".topo-node").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedAgentId = el.dataset.agentId;
      renderAgentDrawer(topo, view);
    });
  });

  if (state.selectedAgentId) renderAgentDrawer(topo, view);
  else document.getElementById("agent-drawer").classList.add("hidden");
}

function renderAgentDrawer(topo, view) {
  const drawer = document.getElementById("agent-drawer");
  const node = (topo.nodes || []).find((n) => n.id === state.selectedAgentId);
  if (!node) {
    drawer.classList.add("hidden");
    return;
  }
  drawer.classList.remove("hidden");
  const tasksHtml = (node.tasks || []).length
    ? node.tasks
        .map(
          (t) => `<div class="sub-box"><strong>${esc(t.task_id)}</strong> [<span class="pill ${statusPillClass(t.status)}">${esc(t.status)}</span>] — ${esc(t.title || t.description)}</div>`
        )
        .join("")
    : `<div class="muted">${view.planRevealed ? "No specialist tasks assigned directly to this component." : "Awaiting plan_created event."}</div>`;

  const msgs = (view.messages || []).filter((m) => m.sender_id === node.id || m.recipient_id === node.id);
  const msgsHtml = msgs.length
    ? `<div><strong>Peer Messages (${msgs.length}):</strong></div>` +
      msgs.map((m) => `<div class="sub-box"><code>${esc(m.sender_id)} → ${esc(m.recipient_id)}</code>: ${esc(m.content)}</div>`).join("")
    : "";

  drawer.innerHTML = `
    <div class="panel-header">
      <h3>Agent Activity: ${esc(node.label)} (${esc(node.id)})</h3>
      <span class="pill ${statusPillClass(node.status)}">${esc(node.status)}</span>
    </div>
    <div>Model: <code>${esc(node.model || "deterministic")}</code></div>
    ${tasksHtml}
    ${msgsHtml}
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
    list.innerHTML = `<div class="muted">No events match this filter at the current cursor.</div>`;
    return;
  }
  list.innerHTML = filtered
    .map((ev) => {
      const ts = String(eventTime(ev)).slice(11, 19) || "--:--:--";
      const msg = ev.message ? `${ev.message} ` : "";
      const payloadStr = ev.payload ? JSON.stringify(ev.payload).slice(0, 140) : "";
      return `
        <div class="ev-row">
          <div class="ev-seq">#${esc(ev.seq)} ${esc(ts)}</div>
          <div class="ev-type">${esc(ev.event_type)}</div>
          <div class="ev-desc">${esc(msg)}<span class="muted">${esc(payloadStr)}</span></div>
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
      renderClaimDetail(view, el.dataset.cid, Number(el.dataset.ver));
    });
  });

  if (state.selectedClaimId) {
    const [cid, ver] = state.selectedClaimId.split("@");
    renderClaimDetail(view, cid, Number(ver));
  } else {
    document.getElementById("claim-detail").classList.add("hidden");
  }
}

function renderClaimDetail(view, claimId, version) {
  const detail = document.getElementById("claim-detail");
  const claim = (view.claims || []).find((c) => c.claim_id === claimId && Number(c.version) === version);
  if (!claim) {
    detail.classList.add("hidden");
    return;
  }
  detail.classList.remove("hidden");

  const evdById = Object.fromEntries((view.evidence || []).map((e) => [e.evidence_id, e]));
  const toolById = Object.fromEntries((view.tools || []).map((t) => [t.tool_execution_id, t]));
  const verdicts = (view.verdicts || []).filter((v) => v.claim_id === claimId);
  const versions = (view.claims || []).filter((c) => c.claim_id === claimId);

  const deps = [
    ...(claim.depends_on_claim_ids || []),
    ...(claim.premise_refs || []),
  ];

  const evdHtml =
    (claim.evidence_ids || [])
      .map((eid) => {
        const e = evdById[eid];
        if (!e) return `<div class="sub-box">Evidence ${esc(eid)} (not yet visible)</div>`;
        return `<div class="sub-box"><strong>${esc(eid)}</strong> [${esc(e.source_type)} · <code>${esc(e.source_ref || "none")}</code> · verified=${esc(e.verified_in_input)}]<br/>${esc(e.excerpt)}</div>`;
      })
      .join("") || `<div class="muted">No supporting evidence linked.</div>`;

  const toolHtml =
    (claim.tool_execution_ids || [])
      .map((tid) => {
        const t = toolById[tid];
        if (!t) return `<div class="sub-box">Tool ${esc(tid)} (not yet visible)</div>`;
        const provKeys = Object.entries(t.provenance_details || {})
          .map(([k, v]) => `${k}→${v.ref} (${v.grounded ? "grounded" : "ungrounded"})`)
          .join(", ");
        return `<div class="sub-box">
          <strong>${esc(t.tool_name)}</strong> (${t.success ? "success" : "failed"}) ·
          expr: <code>${esc(t.expression)}</code> = <strong>${esc(t.computed_result ?? "none")}</strong> ${esc(t.units || "")}<br/>
          <span class="muted">vars: ${esc(JSON.stringify(t.variables || {}))} · grounded=${esc(t.inputs_grounded)} · matches_proposed=${esc(t.matches_proposed)}</span>
          ${provKeys ? `<br/><span class="muted">provenance: ${esc(provKeys)}</span>` : ""}
          ${t.error ? `<br/><span style="color:#fca5a5">error: ${esc(t.error)}</span>` : ""}
        </div>`;
      })
      .join("") || `<div class="muted">No deterministic tool executions linked.</div>`;

  const verdictHtml = verdicts.length
    ? verdicts
        .map((v) => {
          const dec = v.decision || v.verdict || "";
          const fix = v.required_correction || v.required_fix || "";
          return `<div class="sub-box"><span class="pill ${statusPillClass(dec)}">${esc(dec)}</span> <strong>${esc(v.reviewer_id)}</strong> (v${esc(v.claim_version)}): ${esc(v.rationale)}${fix ? ` <em>Fix: ${esc(fix)}</em>` : ""}</div>`;
        })
        .join("")
    : `<div class="muted">No reviewer verdict recorded yet.</div>`;

  detail.innerHTML = `
    <div class="panel-header">
      <h3>${esc(claim.claim_id)} (v${esc(claim.version)})</h3>
      <span class="pill ${statusPillClass(claim.status)}">${esc(claim.status)}</span>
    </div>
    <div><strong>Requirements:</strong> ${esc((claim.requirement_refs || []).join(", ") || "None")}</div>
    <div><strong>Premise Dependencies:</strong> ${esc(deps.join(", ") || "None")}</div>
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
    body.innerHTML = `<div class="muted">Advance replay to the <code>honesty_gate_evaluated</code> event to inspect the release decision and final answer.</div>`;
    return;
  }

  if (!gate) {
    pill.textContent = "NOT RECORDED";
    pill.className = "pill pill-red";
    body.innerHTML = `<div class="muted">No Honesty Gate decision persisted for this run.</div>`;
    return;
  }

  const vs = gate.verification_summary || finalRes?.verification || {};
  const rmode = String(gate.release_mode || "unknown").toLowerCase();
  const pillCls = !gate.passed || rmode === "blocked" ? "pill-red" : rmode === "partial" ? "pill-amber" : "pill-green";
  pill.textContent = `${gate.release_mode} (${gate.passed ? "PASSED" : "BLOCKED"})`;
  pill.className = `pill ${pillCls}`;

  const answerText = finalRes && finalRes.answer ? finalRes.answer : view.runCompleted ? "(No final answer released)" : "(Awaiting run_completed event...)";
  const missing = (finalRes && finalRes.missing_information) || [];
  const reasons = gate.reasons || vs.honesty_gate_reasons || [];
  const rejSegs = gate.rejected_segments || [];
  const rejClaims = gate.rejected_claim_refs || [];
  const reqStatuses = vs.requirement_statuses || [];

  const reqHtml = reqStatuses.length
    ? `<div><strong>Requirement Linkage Details:</strong></div>` +
      reqStatuses
        .map(
          (r) => `<div class="sub-box"><span class="pill ${statusPillClass(r.status)}">${esc(r.status)}</span> <strong>${esc(r.requirement_id)}</strong>: ${esc(r.text)}<br/><span class="muted">${esc(r.rationale)}</span></div>`
        )
        .join("")
    : "";

  body.innerHTML = `
    <div class="banner banner-blue">
      <strong>Release Distinction:</strong>
      release_mode=<code>${esc(gate.release_mode)}</code> ·
      gate_passed=<code>${esc(gate.passed)}</code> ·
      execution_completed=<code>${esc(vs.execution_completed ?? "Unknown")}</code> ·
      content_released_safely=<code>${esc(vs.content_released_safely ?? "Unknown")}</code>.
      Passing the Honesty Gate verifies claim-level provenance and review, <strong>not</strong> factual correctness or complete requirement coverage.
    </div>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Supported Claims</div><div class="kpi-val">${esc(vs.supported_claim_count ?? "Unknown")}</div></div>
      <div class="kpi-card"><div class="kpi-label">Rejected / Withheld</div><div class="kpi-val">${esc(vs.rejected_claim_count ?? rejClaims.length)}</div></div>
      <div class="kpi-card"><div class="kpi-label">Rejected Segments</div><div class="kpi-val">${esc(rejSegs.length)}</div></div>
      <div class="kpi-card"><div class="kpi-label">Unresolved / Partial Reqs</div><div class="kpi-val">${esc(vs.requirements_unresolved ?? "Unknown")} / ${esc(vs.requirements_partial ?? "Unknown")}</div></div>
    </div>
    <div><strong>Gate Reasons:</strong> ${esc(reasons.join(" | ") || "None")}</div>
    <div><strong>Missing Information:</strong> ${esc(missing.join(" | ") || "None")}</div>
    ${reqHtml}
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
  document.getElementById("chk-follow-newest").addEventListener("change", (e) => {
    state.followNewest = Boolean(e.target.checked);
    if (state.followNewest) refreshLiveRuns();
  });
  document.getElementById("btn-refresh-live").addEventListener("click", refreshLiveRuns);
  document.getElementById("live-select").addEventListener("change", (e) => {
    state.followNewest = false;
    document.getElementById("chk-follow-newest").checked = false;
    loadLiveRun(e.target.value);
  });

  setTab("replay");
});
