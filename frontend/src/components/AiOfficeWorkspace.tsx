import React, { useEffect, useRef, useState } from "react";
import { FullBodyCharacter } from "../characters/FullBodyCharacter";
import {
  LEAD_DESK,
  OFFICE_DOOR,
  OFFICE_HEIGHT,
  OFFICE_WIDTH,
  OfficePacket,
  OfficeSimState,
  REVIEW_BAY,
  SPECIALIST_DESKS,
  applyTraceToOffice,
  createInitialOfficeState,
  stepOfficeSimulation,
} from "../office/officeEngine";
import type { TracePayload } from "../types";

interface AiOfficeProps {
  trace: TracePayload | null;
  isRunning: boolean;
  mode: "network" | "baseline";
}

export function isVerificationBlocked(trace: TracePayload | null): boolean {
  if (!trace) return false;
  const status = String(trace.status || "").toLowerCase();
  if (status === "failed" || status === "cancelled") return true;
  const gate = trace.honesty_gate;
  if (gate && (gate.passed === false || String(gate.release_mode).toLowerCase() === "blocked")) {
    return true;
  }
  return false;
}

export function deriveStage(trace: TracePayload | null, isRunning: boolean): {
  step: number;
  blocked: boolean;
  headline: string;
  subline: string;
  leadBubble: string;
} {
  if (!trace) {
    if (isRunning) {
      return {
        step: 1,
        blocked: false,
        headline: "Your AI Team is assembling...",
        subline: "Lead is analyzing the prompt and forming the team.",
        leadBubble: "Analyzing prompt & planning...",
      };
    }
    return {
      step: 0,
      blocked: false,
      headline: "CrewForge AI Office",
      subline: "Ask a question below to watch your dynamic team collaborate in real time.",
      leadBubble: "Ready when you are.",
    };
  }

  const events = trace.execution_events || [];
  const types = new Set(events.map((e) => e.event_type));
  const status = String(trace.status || "").toLowerCase();
  const blocked = isVerificationBlocked(trace);

  if (status === "failed" || types.has("run_failed")) {
    return {
      step: 4,
      blocked: true,
      headline: "Execution Failed",
      subline: "Run failed — inspect the error or Mission Control trace for details.",
      leadBubble: "Execution failed.",
    };
  }
  if ((status === "completed" || status === "partial" || types.has("run_completed")) && blocked) {
    return {
      step: 4,
      blocked: true,
      headline: "Verification Blocked Release",
      subline: "Honesty Gate withheld unverified or rejected claims.",
      leadBubble: "Release blocked by Honesty Gate.",
    };
  }
  if (status === "completed" || status === "partial" || types.has("run_completed")) {
    return {
      step: 4,
      blocked: false,
      headline: status === "completed" ? "Mission Accomplished" : "Completed (Partial Release)",
      subline: `${trace.roster?.length || 1} agents collaborated · ${events.length} persisted events`,
      leadBubble: status === "completed" ? "Verified & delivered final answer!" : "Released verified subset.",
    };
  }
  if (types.has("honesty_gate_evaluated") || types.has("synthesis_completed")) {
    return {
      step: 4,
      blocked,
      headline: "Finalizing Verified Answer...",
      subline: "Honesty Gate is checking claim provenance and release rules.",
      leadBubble: "Running Honesty Gate...",
    };
  }
  if (types.has("review_completed") || types.has("claim_revised")) {
    return {
      step: 3,
      blocked: false,
      headline: "Reviewing Claims & Evidence...",
      subline: "Independent reviewers are verifying arithmetic and provenance.",
      leadBubble: "Checking reviewer verdicts...",
    };
  }
  if (types.has("plan_created") || types.has("task_started")) {
    return {
      step: 2,
      blocked: false,
      headline: "Your AI Team is on it!",
      subline: "Lead has assembled the team and specialists are executing tasks.",
      leadBubble: "Coordinating specialist tasks...",
    };
  }
  return {
    step: 1,
    blocked: false,
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
  const simRef = useRef<OfficeSimState>(sim);
  const rafRef = useRef<number | null>(null);

  const stage = deriveStage(trace, isRunning);
  const events = trace?.execution_events || [];

  const elapsed =
    trace?.latency_ms !== null && trace?.latency_ms !== undefined
      ? `${(trace.latency_ms / 1000).toFixed(1)}s`
      : isRunning
      ? "Live"
      : "—";

  // Ingest new trace updates (handling batched events cleanly)
  useEffect(() => {
    const t = performance.now();
    const updated = applyTraceToOffice(simRef.current, trace, isRunning, t, stage.blocked);
    simRef.current = updated;
    setSim(updated);
    setNowMs(t);
  }, [trace, isRunning, stage.blocked]);

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
                : stage.blocked
                ? "pill-error"
                : "pill-idle"
            }`}
          >
            <span className="pulse-dot" />
            {isRunning
              ? trace?.waiting_for_provider
                ? "Waiting for Provider"
                : "Live Execution"
              : stage.blocked
              ? "Verification Blocked"
              : trace
              ? String(trace.status).toUpperCase()
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
          <div className="office-2d-viewport">
            <svg
              viewBox={`0 0 ${OFFICE_WIDTH} ${OFFICE_HEIGHT}`}
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
              <rect x="0" y="0" width={OFFICE_WIDTH} height="132" fill="url(#cf-wall-grad)" />
              {/* Wall trim */}
              <rect x="0" y="128" width={OFFICE_WIDTH} height="6" fill="#4338ca" fillOpacity="0.6" />

              {/* Panoramic Windows */}
              <g>
                <rect x="235" y="16" width="160" height="82" rx="6" fill="url(#cf-win-grad)" stroke="#6366f1" strokeOpacity="0.45" strokeWidth="2" />
                <rect x="565" y="16" width="160" height="82" rx="6" fill="url(#cf-win-grad)" stroke="#6366f1" strokeOpacity="0.45" strokeWidth="2" />
                {/* Subtle skyline silhouettes */}
                <path d="M245 96 V58 H268 V70 H290 V44 H316 V74 H345 V52 H372 V96 Z" fill="#312e81" fillOpacity="0.65" />
                <path d="M575 96 V62 H600 V48 H628 V68 H656 V42 H686 V66 H715 V96 Z" fill="#312e81" fillOpacity="0.65" />
                <circle cx="360" cy="34" r="5" fill="#fde68a" fillOpacity="0.7" />
                <circle cx="695" cy="32" r="4" fill="#fde68a" fillOpacity="0.6" />
              </g>

              {/* Center CrewForge Sign */}
              <g transform="translate(480, 48)">
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

              {/* Right Bookshelf & Watercooler */}
              <g transform="translate(760, 22)">
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
              <rect x="0" y="134" width={OFFICE_WIDTH} height={OFFICE_HEIGHT - 134} fill="url(#cf-floor-grad)" />
              {/* Subtle floorboard lines */}
              {[180, 230, 280, 330, 380].map((fy) => (
                <line
                  key={fy}
                  x1="0"
                  y1={fy}
                  x2={OFFICE_WIDTH}
                  y2={fy}
                  stroke="#ffffff"
                  strokeOpacity="0.04"
                  strokeWidth="1"
                />
              ))}

              {/* Office Entrance Door (where dynamic specialists walk in) */}
              <g transform={`translate(${OFFICE_DOOR.x - 24}, ${OFFICE_DOOR.y - 75})`}>
                <rect x="0" y="0" width="44" height="82" rx="4" fill="#475569" stroke="#94a3b8" strokeWidth="1.5" />
                <rect x="6" y="8" width="32" height="28" rx="2" fill="#38bdf8" fillOpacity="0.2" />
                <circle cx="36" cy="48" r="2.5" fill="#fbbf24" />
                <text x="22" y="95" textAnchor="middle" fontSize="8.5" fill="#94a3b8">
                  Entrance
                </text>
              </g>

              {/* Potted Plants */}
              {[
                { x: 910, y: 155 },
                { x: 88, y: 390 },
                { x: 915, y: 390 },
              ].map((p, i) => (
                <g key={i} transform={`translate(${p.x}, ${p.y})`}>
                  <ellipse cx="0" cy="-16" rx="14" ry="12" fill="#15803d" />
                  <ellipse cx="-6" cy="-22" rx="8" ry="8" fill="#22c55e" />
                  <ellipse cx="6" cy="-20" rx="8" ry="8" fill="#16a34a" />
                  <polygon points="-9,-6 9,-6 7,10 -7,10" fill="#b45309" />
                </g>
              ))}

              {/* Desks & Workstations */}
              {renderDesk(
                LEAD_DESK.x,
                LEAD_DESK.y + 16,
                "#6366f1",
                sim.actors.lead?.pose === "working",
                "Lead Command",
                "desk_lead"
              )}
              {SPECIALIST_DESKS.slice(0, 3).map((d, idx) =>
                renderDesk(d.x, d.y + 18, "#38bdf8", false, `Workstation ${idx + 1}`, `desk_sp_${idx}`)
              )}
              {renderDesk(
                REVIEW_BAY.x,
                REVIEW_BAY.y + 18,
                "#10b981",
                sim.actors.reviewer?.pose === "reviewing",
                "Review & Gate",
                "desk_review"
              )}

              {/* Depth-sorted Full-Body Characters */}
              {sortedActors.map((actor) => (
                <FullBodyCharacter
                  key={actor.id}
                  profile={actor.profile}
                  pose={actor.pose}
                  x={actor.x}
                  y={actor.y}
                  facing={actor.facing}
                  roleLabel={actor.role}
                  bubble={actor.bubble}
                />
              ))}

              {/* Animated Document Handovers, Envelopes, and Revisions */}
              {sim.packets.map((pkt) => renderPacket(pkt, nowMs))}
            </svg>
          </div>

          <div className="office-ticker" aria-live="polite">
            <span className="ticker-badge">Office Feed</span>
            <span className="ticker-text">{sim.lastInteractionNote}</span>
            <span className="ticker-meta">
              {Object.keys(sim.actors).length} active in room · {events.length} events
            </span>
          </div>
        </>
      )}
    </section>
  );
};
