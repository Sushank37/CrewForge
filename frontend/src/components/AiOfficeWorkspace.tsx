import React, { useEffect, useMemo, useRef, useState } from "react";
import { FullBodyCharacter } from "../characters/FullBodyCharacter";
import {
  OFFICE_DOOR,
  OfficePacket,
  OfficeSimState,
  applyTraceToOffice,
  classifyReleaseOutcome,
  computeOfficeLayout,
  createInitialOfficeState,
  stepOfficeSimulation,
} from "../office/officeEngine";
import type { OperationalState, ReleaseOutcome } from "../office/officeEngine";
import type { TracePayload } from "../types";

interface AiOfficeProps {
  trace: TracePayload | null;
  isRunning: boolean;
  mode: "network" | "baseline";
}

export function isVerificationBlocked(trace: TracePayload | null): boolean {
  return classifyReleaseOutcome(trace) === "blocked";
}

export function deriveStage(trace: TracePayload | null, isRunning: boolean): {
  step: number;
  blocked: boolean;
  releaseOutcome: ReleaseOutcome;
  headline: string;
  subline: string;
  leadBubble: string;
} {
  if (!trace) {
    if (isRunning) {
      return {
        step: 1,
        blocked: false,
        releaseOutcome: "none",
        headline: "Your AI Team is assembling...",
        subline: "Lead is analyzing the prompt and forming the team.",
        leadBubble: "Analyzing prompt & planning...",
      };
    }
    return {
      step: 0,
      blocked: false,
      releaseOutcome: "none",
      headline: "CrewForge AI Office",
      subline: "Ask a question below to watch your dynamic team collaborate in real time.",
      leadBubble: "Ready when you are.",
    };
  }

  const events = trace.execution_events || [];
  const types = new Set(events.map((e) => e.event_type));
  const status = String(trace.status || "").toLowerCase();
  const releaseOutcome = classifyReleaseOutcome(trace);
  const blocked = releaseOutcome === "blocked";

  if (status === "failed" || types.has("run_failed")) {
    return {
      step: 4,
      blocked: true,
      releaseOutcome: "blocked",
      headline: "Execution Failed",
      subline: "Run failed — inspect the error or Mission Control trace for details.",
      leadBubble: "Execution failed.",
    };
  }
  if ((status === "completed" || status === "partial" || types.has("run_completed")) && blocked) {
    return {
      step: 4,
      blocked: true,
      releaseOutcome: "blocked",
      headline: "Verification Blocked Release",
      subline: "Honesty Gate blocked release; no verified claims could be delivered.",
      leadBubble: "Release blocked by Honesty Gate.",
    };
  }
  if (status === "completed" || status === "partial" || types.has("run_completed")) {
    if (String(trace.mode || "").toLowerCase() === "baseline") {
      return {
        step: 4,
        blocked: false,
        releaseOutcome: "none",
        headline: "Baseline Complete (Unverified)",
        subline: "Single-agent baseline finished without independent claim verification.",
        leadBubble: "Baseline output ready (unverified).",
      };
    }
    const isPartial = releaseOutcome === "partial";
    return {
      step: 4,
      blocked: false,
      releaseOutcome,
      headline: isPartial ? "Partial Verified Release" : "Mission Accomplished",
      subline: isPartial
        ? `${trace.roster?.length || 1} agents collaborated · verified subset released, unverified claims withheld`
        : `${trace.roster?.length || 1} agents collaborated · ${events.length} persisted events`,
      leadBubble: isPartial ? "Released verified subset." : "Verified & delivered final answer!",
    };
  }
  if (types.has("honesty_gate_evaluated") || types.has("synthesis_completed")) {
    return {
      step: 4,
      blocked,
      releaseOutcome,
      headline: "Finalizing Verified Answer...",
      subline: "Honesty Gate is checking claim provenance and release rules.",
      leadBubble: "Running Honesty Gate...",
    };
  }
  if (types.has("review_completed") || types.has("claim_revised")) {
    return {
      step: 3,
      blocked: false,
      releaseOutcome: "none",
      headline: "Reviewing Claims & Evidence...",
      subline: "Independent reviewers are verifying arithmetic and provenance.",
      leadBubble: "Checking reviewer verdicts...",
    };
  }
  if (types.has("plan_created") || types.has("task_started")) {
    return {
      step: 2,
      blocked: false,
      releaseOutcome: "none",
      headline: "Your AI Team is on it!",
      subline: "Lead has assembled the team and specialists are executing tasks.",
      leadBubble: "Coordinating specialist tasks...",
    };
  }
  return {
    step: 1,
    blocked: false,
    releaseOutcome: "none",
    headline: "Your AI Team is assembling...",
    subline: "Lead is decomposing the problem into specialist tasks.",
    leadBubble: "Planning DAG & specialists...",
  };
}

function renderDesk(
  x: number,
  y: number,
  accent: string,
  active: boolean,
  label: string,
  key: string
) {
  return (
    <g key={key} transform={`translate(${x}, ${y})`}>
      {/* Ambient desk glow when active */}
      {active && (
        <ellipse cx="0" cy="6" rx="58" ry="20" fill={accent} fillOpacity="0.14" />
      )}
      {/* Chair back behind desk */}
      <rect x="-14" y="-8" width="28" height="14" rx="5" fill="#1e293b" stroke="#334155" strokeWidth="1" />
      {/* Desk surface */}
      <rect x="-52" y="2" width="104" height="24" rx="5" fill="#7c4a2d" stroke="#452312" strokeWidth="1.5" />
      <rect x="-52" y="22" width="104" height="7" rx="2" fill="#5c331c" />
      {/* Desk legs */}
      <rect x="-46" y="28" width="6" height="12" fill="#3b2011" />
      <rect x="40" y="28" width="6" height="12" fill="#3b2011" />
      {/* Monitor / Laptop */}
      <rect
        x="-16"
        y="-7"
        width="32"
        height="18"
        rx="2.5"
        fill="#0f172a"
        stroke={active ? accent : "#475569"}
        strokeWidth="1.5"
      />
      <rect
        x="-13"
        y="-4"
        width="26"
        height="12"
        rx="1.5"
        fill={active ? accent : "#1e293b"}
        fillOpacity={active ? 0.35 : 0.9}
      />
      {/* Coffee mug */}
      <rect x="26" y="4" width="7" height="7" rx="1.5" fill="#e2e8f0" />
      <rect x="-38" y="6" width="14" height="5" rx="1" fill="#334155" />
      {/* Desk placard */}
      <text x="0" y="19" textAnchor="middle" fontSize="8" fontWeight="600" fill="#fde68a" fillOpacity="0.72">
        {label}
      </text>
    </g>
  );
}

function renderPacket(pkt: OfficePacket, nowMs: number) {
  const rawT = (nowMs - pkt.startTime) / pkt.durationMs;
  if (rawT < 0 || rawT > 1) return null;
  const t = rawT < 0.5 ? 2 * rawT * rawT : 1 - Math.pow(-2 * rawT + 2, 2) / 2;
  const ctrlX = (pkt.fromX + pkt.toX) / 2;
  const ctrlY = Math.min(pkt.fromY, pkt.toY) - 42;
  const inv = 1 - t;
  const x = inv * inv * pkt.fromX + 2 * inv * t * ctrlX + t * t * pkt.toX;
  const y = inv * inv * pkt.fromY + 2 * inv * t * ctrlY + t * t * pkt.toY;

  return (
    <g key={pkt.id} transform={`translate(${x}, ${y})`}>
      <path
        d={`M ${pkt.fromX - x} ${pkt.fromY - y} Q ${ctrlX - x} ${ctrlY - y} 0 0`}
        stroke={pkt.color}
        strokeWidth="1.6"
        strokeDasharray="4 4"
        strokeOpacity="0.55"
        fill="none"
      />
      <rect
        x="-18"
        y="-10"
        width="36"
        height="20"
        rx="5"
        fill="#0f172a"
        stroke={pkt.color}
        strokeWidth="1.6"
      />
      {pkt.kind === "message" ? (
        <path d="M-12 -5 L0 2 L12 -5" stroke={pkt.color} strokeWidth="1.4" fill="none" />
      ) : (
        <text x="0" y="3.5" textAnchor="middle" fontSize="8.5" fontWeight="700" fill="#f8fafc">
          {pkt.label}
        </text>
      )}
    </g>
  );
}

export const AiOfficeWorkspace: React.FC<AiOfficeProps> = ({ trace, isRunning, mode }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [sim, setSim] = useState<OfficeSimState>(() => createInitialOfficeState());
  const [nowMs, setNowMs] = useState<number>(() => performance.now());
  const [wallNowMs, setWallNowMs] = useState<number>(() => Date.now());
  const [selectedActorId, setSelectedActorId] = useState<string | null>(null);
  const [showTimeline, setShowTimeline] = useState<boolean>(true);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const simRef = useRef<OfficeSimState>(sim);
  const rafRef = useRef<number | null>(null);

  const stage = deriveStage(trace, isRunning);
  const events = trace?.execution_events || [];
  const specialists = useMemo(
    () => (trace?.roster || []).filter((a) => a.role.toLowerCase() !== "lead"),
    [trace?.roster]
  );
  const layout = useMemo(() => computeOfficeLayout(specialists.length), [specialists.length]);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [trace?.run_id, specialists.length]);

  useEffect(() => {
    if (!isRunning) return;
    const id = window.setInterval(() => {
      setWallNowMs(Date.now());
    }, 1000);
    return () => window.clearInterval(id);
  }, [isRunning]);

  const elapsed = (() => {
    if (trace?.latency_ms !== null && trace?.latency_ms !== undefined) {
      return `${(trace.latency_ms / 1000).toFixed(1)}s`;
    }
    if (isRunning && trace?.created_at) {
      const start = Date.parse(trace.created_at);
      if (!Number.isNaN(start)) {
        return `${Math.max(0, (wallNowMs - start) / 1000).toFixed(0)}s`;
      }
    }
    return isRunning ? "Live" : "—";
  })();

  const stateBadgeLabel = (st?: OperationalState): string => {
    switch (st) {
      case "active_inference":
        return "Active Inference";
      case "dependency_waiting":
        return "Dependency Wait";
      case "quota_waiting":
        return "Quota Wait";
      case "reviewing":
        return "Reviewing";
      case "revising":
        return "Revising";
      case "failed":
        return "Failed";
      case "completed":
        return "Completed";
      default:
        return "Idle";
    }
  };

  const vbWidth = layout.width / zoom;
  const vbHeight = layout.height / zoom;
  const vbX = (layout.width - vbWidth) / 2 - pan.x;
  const vbY = (layout.height - vbHeight) / 2 - pan.y;
  const selectedActor = selectedActorId ? sim.actors[selectedActorId] || null : null;

  // Ingest new trace updates (handling batched events cleanly)
  useEffect(() => {
    const t = performance.now();
    const updated = applyTraceToOffice(simRef.current, trace, isRunning, t);
    simRef.current = updated;
    setSim(updated);
    setNowMs(t);
  }, [trace, isRunning, stage.releaseOutcome]);

  // Lightweight requestAnimationFrame loop (auto-pauses when motion settles)
  useEffect(() => {
    let lastTs = performance.now();
    let active = true;

    const tick = (ts: number) => {
      if (!active) return;
      const dt = Math.min(0.05, Math.max(0.001, (ts - lastTs) / 1000));
      lastTs = ts;
      const { next, hasMotion } = stepOfficeSimulation(simRef.current, dt, ts);
      simRef.current = next;
      setSim(next);
      setNowMs(ts);
      if (hasMotion || isRunning) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        rafRef.current = null;
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      active = false;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [trace, isRunning]);

  const sortedActors = Object.values(sim.actors).sort((a, b) => a.y - b.y);

  return (
    <section
      className={`office-shell ${collapsed ? "office-collapsed" : ""}`}
      aria-label="AI Office Workspace"
    >
      <div className="office-header">
        <div className="office-title-block">
          <div className="office-headline">
            {isRunning && trace?.waiting_for_provider
              ? "Waiting for Provider Quota..."
              : stage.headline}
          </div>
          <div className="office-subline">
            {isRunning && trace?.waiting_for_provider
              ? trace.waiting_reason || "Waiting for provider rate-limit reset..."
              : stage.subline}
          </div>
        </div>
        <div className="office-header-actions">
          <span
            className={`office-pill ${
              isRunning
                ? trace?.waiting_for_provider
                  ? "pill-warn"
                  : "pill-live"
                : stage.releaseOutcome === "blocked"
                ? "pill-error"
                : stage.releaseOutcome === "partial"
                ? "pill-warn"
                : stage.releaseOutcome === "full"
                ? "pill-success"
                : "pill-idle"
            }`}
          >
            <span className="pulse-dot" />
            {isRunning
              ? trace?.waiting_for_provider
                ? "Waiting for Provider"
                : "Live Execution"
              : stage.releaseOutcome === "blocked"
              ? "Release Blocked"
              : stage.releaseOutcome === "partial"
              ? "Partial Verified Release"
              : stage.releaseOutcome === "full"
              ? "Full Verified Release"
              : mode === "baseline"
              ? "Baseline"
              : "Standby"}
            {elapsed !== "—" ? ` · ${elapsed}` : ""}
          </span>
          <button
            type="button"
            className="btn-ghost btn-sm"
            onClick={() => setCollapsed((c) => !c)}
          >
            {collapsed ? "Expand Office" : "Collapse"}
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          <div className="office-controls">
            <div className="office-controls-left">
              <span className="muted" style={{ fontSize: 11.5 }}>
                {specialists.length} specialist(s) · {layout.rows} row(s) · {trace?.usage?.model_calls || 0} model call(s)
              </span>
              {selectedActor && (
                <button
                  type="button"
                  className="btn-ghost btn-xs"
                  onClick={() => setSelectedActorId(null)}
                >
                  Clear selection
                </button>
              )}
            </div>
            <div className="office-controls-right">
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={() => setShowTimeline((v) => !v)}
              >
                {showTimeline ? "Hide Timeline" : "Show Timeline"}
              </button>
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={() => setZoom((z) => Math.max(0.5, Number((z - 0.15).toFixed(2))))}
                title="Zoom out"
              >
                −
              </button>
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={() => {
                  setZoom(1);
                  setPan({ x: 0, y: 0 });
                }}
                title="Fit to view"
              >
                Fit ({Math.round(zoom * 100)}%)
              </button>
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={() => setZoom((z) => Math.min(2.2, Number((z + 0.15).toFixed(2))))}
                title="Zoom in"
              >
                +
              </button>
            </div>
          </div>

          <div
            className="office-2d-viewport"
            onWheel={(e) => {
              if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > 0) {
                const factor = e.deltaY > 0 ? -0.08 : 0.08;
                setZoom((z) => Math.min(2.2, Math.max(0.5, Number((z + factor).toFixed(2)))));
              }
            }}
            onPointerDown={(e) => {
              if ((e.target as HTMLElement).closest("[data-actor-id]")) return;
              dragRef.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
            }}
            onPointerMove={(e) => {
              if (!dragRef.current) return;
              const dx = (e.clientX - dragRef.current.x) / zoom;
              const dy = (e.clientY - dragRef.current.y) / zoom;
              setPan({ x: dragRef.current.panX + dx, y: dragRef.current.panY + dy });
            }}
            onPointerUp={() => {
              dragRef.current = null;
            }}
            onPointerLeave={() => {
              dragRef.current = null;
            }}
          >
            <svg
              viewBox={`${vbX} ${vbY} ${vbWidth} ${vbHeight}`}
              className="office-2d-svg"
              role="img"
              aria-label="CrewForge 2D Miniature AI Office"
            >
              <defs>
                <linearGradient id="cf-wall-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#1e1b4b" />
                  <stop offset="100%" stopColor="#312e81" />
                </linearGradient>
                <linearGradient id="cf-floor-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#292038" />
                  <stop offset="100%" stopColor="#1b1528" />
                </linearGradient>
                <linearGradient id="cf-win-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0f172a" />
                  <stop offset="100%" stopColor="#1e293b" />
                </linearGradient>
              </defs>

              {/* Back Wall */}
              <rect x="0" y="0" width={layout.width} height="132" fill="url(#cf-wall-grad)" />
              <rect x="0" y="128" width={layout.width} height="6" fill="#4338ca" fillOpacity="0.6" />

              {/* Panoramic Windows */}
              <g>
                <rect x="235" y="16" width="160" height="82" rx="6" fill="url(#cf-win-grad)" stroke="#6366f1" strokeOpacity="0.45" strokeWidth="2" />
                <rect x={layout.width - 395} y="16" width="160" height="82" rx="6" fill="url(#cf-win-grad)" stroke="#6366f1" strokeOpacity="0.45" strokeWidth="2" />
                <circle cx="360" cy="34" r="5" fill="#fde68a" fillOpacity="0.7" />
                <circle cx={layout.width - 265} cy="32" r="4" fill="#fde68a" fillOpacity="0.6" />
              </g>

              {/* Center CrewForge Sign */}
              <g transform={`translate(${layout.leadDesk.x}, 48)`}>
                <rect x="-72" y="-22" width="144" height="44" rx="10" fill="#0f172a" fillOpacity="0.85" stroke="#818cf8" strokeWidth="1.6" />
                <text x="0" y="6" textAnchor="middle" fontSize="18" fontWeight="800" fill="#e0e7ff">
                  CrewForge
                </text>
              </g>

              {/* Left Wall Execution Whiteboard */}
              <g transform="translate(125, 20)">
                <rect x="0" y="0" width="90" height="90" rx="5" fill="#f8fafc" stroke="#94a3b8" strokeWidth="2.5" />
                <text x="10" y="17" fontSize="9.5" fontWeight="700" fill="#1e293b">
                  Workflow
                </text>
                {[
                  { label: "1. Plan", done: stage.step >= 1 },
                  { label: "2. Assign", done: stage.step >= 2 },
                  { label: "3. Work", done: stage.step >= 2 },
                  { label: "4. Review", done: stage.step >= 3 },
                  { label: "5. Finalize", done: stage.step >= 4 },
                ].map((item, idx) => (
                  <text
                    key={item.label}
                    x="10"
                    y={32 + idx * 12.5}
                    fontSize="8.8"
                    fontWeight="600"
                    fill={item.done ? "#059669" : "#64748b"}
                  >
                    {item.done ? "✓ " : "○ "}
                    {item.label}
                  </text>
                ))}
              </g>

              {/* Right Bookshelf */}
              <g transform={`translate(${layout.width - 200}, 22)`}>
                <rect x="0" y="0" width="76" height="96" rx="4" fill="#5c331c" stroke="#3b2011" strokeWidth="1.5" />
                <line x1="4" y1="30" x2="72" y2="30" stroke="#3b2011" strokeWidth="2" />
                <line x1="4" y1="62" x2="72" y2="62" stroke="#3b2011" strokeWidth="2" />
                <rect x="10" y="10" width="10" height="19" fill="#6366f1" />
                <rect x="22" y="12" width="9" height="17" fill="#10b981" />
                <rect x="34" y="8" width="11" height="21" fill="#f59e0b" />
                <rect x="14" y="42" width="12" height="19" fill="#ec4899" />
                <rect x="29" y="44" width="10" height="17" fill="#38bdf8" />
              </g>

              {/* Floor */}
              <rect x="0" y="134" width={layout.width} height={layout.height - 134} fill="url(#cf-floor-grad)" />
              {Array.from({ length: Math.floor((layout.height - 150) / 55) }).map((_, i) => {
                const fy = 180 + i * 55;
                return (
                  <line
                    key={fy}
                    x1="0"
                    y1={fy}
                    x2={layout.width}
                    y2={fy}
                    stroke="#ffffff"
                    strokeOpacity="0.04"
                    strokeWidth="1"
                  />
                );
              })}

              {/* Department / Row Zone Labels when expanded beyond 1 row */}
              {layout.rows > 1 &&
                Array.from({ length: layout.rows }).map((_, r) => (
                  <text
                    key={`wing_${r}`}
                    x="22"
                    y={275 + r * 132}
                    fontSize="10"
                    fontWeight="700"
                    fill="#94a3b8"
                    fillOpacity="0.7"
                  >
                    {r === 0 ? "Ops Bay A" : `Specialist Wing ${String.fromCharCode(65 + r)}`}
                  </text>
                ))}

              {/* Office Entrance Door */}
              <g transform={`translate(${OFFICE_DOOR.x - 24}, ${OFFICE_DOOR.y - 75})`}>
                <rect x="0" y="0" width="44" height="82" rx="4" fill="#475569" stroke="#94a3b8" strokeWidth="1.5" />
                <rect x="6" y="8" width="32" height="28" rx="2" fill="#38bdf8" fillOpacity="0.2" />
                <circle cx="36" cy="48" r="2.5" fill="#fbbf24" />
                <text x="22" y="95" textAnchor="middle" fontSize="8.5" fill="#94a3b8">
                  Entrance
                </text>
              </g>

              {/* Desks & Dynamic Workstations */}
              {renderDesk(
                layout.leadDesk.x,
                layout.leadDesk.y + 16,
                "#6366f1",
                sim.actors.lead?.pose === "working",
                "Lead Command",
                "desk_lead"
              )}
              {layout.desks.map((d, idx) => {
                const sp = specialists[idx];
                const actor = sp ? sim.actors[sp.agent_id] : undefined;
                const isActive = actor?.operationalState === "active_inference";
                return renderDesk(
                  d.x,
                  d.y + 18,
                  "#38bdf8",
                  Boolean(isActive),
                  sp ? sp.role : `Workstation ${idx + 1}`,
                  `desk_sp_${idx}`
                );
              })}
              {renderDesk(
                layout.reviewBay.x,
                layout.reviewBay.y + 18,
                "#10b981",
                sim.actors.reviewer?.pose === "reviewing",
                "Review & Gate",
                "desk_review"
              )}

              {/* Clickable Depth-sorted Full-Body Characters */}
              {sortedActors.map((actor) => {
                const isSel = selectedActorId === actor.id;
                return (
                  <g
                    key={actor.id}
                    data-actor-id={actor.id}
                    style={{ cursor: "pointer" }}
                    onClick={() => setSelectedActorId((prev) => (prev === actor.id ? null : actor.id))}
                  >
                    {isSel && (
                      <ellipse
                        cx={actor.x}
                        cy={actor.y + 22}
                        rx="28"
                        ry="9"
                        fill="none"
                        stroke="#fbbf24"
                        strokeWidth="2.2"
                      />
                    )}
                    <FullBodyCharacter
                      profile={actor.profile}
                      pose={actor.pose}
                      x={actor.x}
                      y={actor.y}
                      facing={actor.facing}
                      roleLabel={actor.role}
                      bubble={actor.bubble}
                    />
                  </g>
                );
              })}

              {/* Animated Document Handovers, Envelopes, and Revisions */}
              {sim.packets.map((pkt) => renderPacket(pkt, nowMs))}
            </svg>
          </div>

          <div className="office-ticker" aria-live="polite">
            <span className="ticker-badge">Office Feed</span>
            <span className="ticker-text">{sim.lastInteractionNote}</span>
            <span className="ticker-meta">
              {Object.keys(sim.actors).length} active in room · {events.length} events · click any agent to inspect
            </span>
          </div>

          {selectedActor && (
            <div className="office-inspector">
              <div className="office-inspector-head">
                <div>
                  <strong>{selectedActor.role}</strong>{" "}
                  <code style={{ fontSize: 11 }}>{selectedActor.id}</code>
                </div>
                <span className={`state-badge state-${selectedActor.operationalState || "idle"}`}>
                  {stateBadgeLabel(selectedActor.operationalState)}
                </span>
              </div>
              <div className="muted" style={{ fontSize: 12 }}>
                Current operation: {selectedActor.currentOperation || "Idle"}
              </div>
              {(() => {
                const tasks = (trace?.tasks || []).filter(
                  (t) => t.assigned_agent_id === selectedActor.id
                );
                const claims = (trace?.claims || []).filter(
                  (c) => c.author_agent_id === selectedActor.id
                );
                const verdicts =
                  selectedActor.id === "reviewer" ? trace?.review_verdicts || [] : [];
                return (
                  <div className="office-inspector-grid">
                    {tasks.length > 0 && (
                      <div>
                        <strong>Tasks ({tasks.length}):</strong>{" "}
                        {tasks.map((t) => `${t.title || t.task_id} [${t.status}]`).join(" · ")}
                      </div>
                    )}
                    {claims.length > 0 && (
                      <div>
                        <strong>Claims ({claims.length}):</strong>{" "}
                        {claims.map((c) => `${c.claim_id}@v${c.version} (${c.status})`).join(" · ")}
                      </div>
                    )}
                    {verdicts.length > 0 && (
                      <div>
                        <strong>Verdicts ({verdicts.length}):</strong>{" "}
                        {verdicts
                          .slice(-4)
                          .map((v) => `${v.claim_id}: ${v.decision || v.verdict}`)
                          .join(" · ")}
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}

          {showTimeline && events.length > 0 && (
            <div className="office-timeline" aria-label="Live Activity Timeline">
              {events
                .slice(-12)
                .reverse()
                .map((ev, idx) => (
                  <div key={ev.event_id || `${ev.seq}_${idx}`} className="office-timeline-row">
                    <span className="timeline-seq">#{ev.seq}</span>
                    <span className="timeline-type">{ev.event_type}</span>
                    <span className="timeline-actor">{ev.actor || "system"}</span>
                    <span className="timeline-msg">{ev.message || ""}</span>
                  </div>
                ))}
            </div>
          )}
        </>
      )}
    </section>
  );
};
