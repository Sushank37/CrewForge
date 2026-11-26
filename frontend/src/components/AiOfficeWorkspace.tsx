import React, { useState } from "react";
import {
  CHARACTER_ROSTER,
  CharacterAvatar,
  CharacterMood,
  pickSpecialistCharacter,
  pickVerifierCharacter,
} from "../characters/CharacterSet";
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

export const AiOfficeWorkspace: React.FC<AiOfficeProps> = ({ trace, isRunning, mode }) => {
  const [collapsed, setCollapsed] = useState(false);

  const stage = deriveStage(trace, isRunning);
  const leadProfile = CHARACTER_ROSTER[0];
  const specialists = (trace?.roster || []).filter((a) => a.role.toLowerCase() !== "lead");
  const tasks = trace?.tasks || [];
  const claims = trace?.claims || [];
  const verdicts = trace?.review_verdicts || [];
  const events = trace?.execution_events || [];

  const elapsed =
    trace?.latency_ms !== null && trace?.latency_ms !== undefined
      ? `${(trace.latency_ms / 1000).toFixed(1)}s`
      : isRunning
      ? "Live"
      : "—";

  const leadMood: CharacterMood =
    stage.blocked
      ? "error"
      : stage.step === 4
      ? "completed"
      : isRunning || stage.step > 0
      ? "working"
      : "idle";

  const lastReviewerId =
    verdicts.length > 0 ? verdicts[verdicts.length - 1].reviewer_id : "reviewer_evidence";
  const verifierProfile = pickVerifierCharacter(lastReviewerId);
  const hasRejectedVerdict = verdicts.some((v) => {
    const d = String(v.decision || v.verdict || "").toLowerCase();
    return d === "reject" || d === "needs_revision";
  });
  const verifierMood: CharacterMood =
    stage.blocked
      ? "error"
      : verdicts.length > 0
      ? stage.step === 4 && !hasRejectedVerdict
        ? "completed"
        : "reviewing"
      : stage.step === 3
      ? "reviewing"
      : "idle";

  const latestEventForAgent = (agentId: string): string | null => {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const ev = events[i];
      if (ev.actor === agentId && ev.message) return ev.message;
    }
    return null;
  };

  return (
    <section className={`office-shell ${collapsed ? "office-collapsed" : ""}`} aria-label="AI Office Workspace">
      <div className="office-header">
        <div className="office-title-block">
          <div className="office-headline">{stage.headline}</div>
          <div className="office-subline">{stage.subline}</div>
        </div>
        <div className="office-header-actions">
          <span className={`office-pill ${isRunning ? "pill-live" : stage.blocked ? "pill-error" : "pill-idle"}`}>
            <span className="pulse-dot" />
            {isRunning
              ? "Live Execution"
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
          <div className="office-stage">
            <div className="office-whiteboard" aria-label="Execution Progress Board">
              <div className="wb-row">
                <span className={stage.step >= 1 ? "wb-done" : "wb-todo"}>
                  {stage.step >= 1 ? "✓" : "○"} Plan
                </span>
              </div>
              <div className="wb-row">
                <span className={stage.step >= 2 ? "wb-done" : "wb-todo"}>
                  {stage.step >= 2 ? "✓" : "○"} Assign
                </span>
              </div>
              <div className="wb-row">
                <span className={stage.step >= 3 ? "wb-done" : stage.step === 2 ? "wb-active" : "wb-todo"}>
                  {stage.step >= 3 ? "✓" : "●"} Execute
                </span>
              </div>
              <div className="wb-row">
                <span className={stage.blocked ? "wb-error" : stage.step >= 4 ? "wb-done" : stage.step === 3 ? "wb-active" : "wb-todo"}>
                  {stage.blocked ? "✕" : stage.step >= 4 ? "✓" : "○"} Review
                </span>
              </div>
              <div className="wb-row">
                <span className={stage.blocked ? "wb-error" : stage.step >= 4 ? "wb-done" : "wb-todo"}>
                  {stage.blocked ? "✕" : stage.step >= 4 ? "✓" : "○"} Finalize
                </span>
              </div>
            </div>

            <div className="office-lead-row">
              <div className={`desk-card desk-lead ${stage.blocked ? "desk-error" : ""}`}>
                <div className="speech-bubble">{stage.leadBubble}</div>
                <CharacterAvatar profile={leadProfile} mood={leadMood} size={68} />
                <div className="desk-name">Lead</div>
                <div className="desk-role">Orchestrator</div>
              </div>
            </div>

            {specialists.length > 0 && (
              <svg className="office-wires" viewBox="0 0 600 32" preserveAspectRatio="none" aria-hidden="true">
                <path
                  d="M300 0 L300 14 M110 14 L490 14 M110 14 L110 32 M300 14 L300 32 M490 14 L490 32"
                  stroke={stage.blocked ? "#ef4444" : "#60a5fa"}
                  strokeWidth="2"
                  fill="none"
                  strokeDasharray={isRunning ? "6 4" : undefined}
                />
              </svg>
            )}

            <div className="office-desks-row">
              {specialists.length === 0 && (
                <div className="desk-card desk-empty">
                  <div className="desk-role">
                    {mode === "baseline"
                      ? "Single-agent Baseline mode (no dynamic specialists)"
                      : isRunning
                      ? "Assembling specialist roles..."
                      : "Dynamic specialists appear here when Lead builds a plan"}
                  </div>
                </div>
              )}

              {specialists.map((spec, idx) => {
                const profile = pickSpecialistCharacter(spec.agent_id, idx);
                const specTask = tasks.find((t) => t.assigned_agent_id === spec.agent_id);
                const tStatus = String(specTask?.status || "pending").toLowerCase();
                const specClaims = claims.filter((c) => c.author_agent_id === spec.agent_id);
                const allRejectedOrStale =
                  specClaims.length > 0 &&
                  specClaims.every((c) => c.status === "rejected" || c.status === "stale") &&
                  !isRunning;

                const mood: CharacterMood =
                  tStatus === "failed" || tStatus === "blocked" || allRejectedOrStale
                    ? "error"
                    : tStatus === "completed"
                    ? "completed"
                    : tStatus === "running" || tStatus === "in_progress"
                    ? "working"
                    : isRunning
                    ? "working"
                    : "idle";
                const bubble = allRejectedOrStale
                  ? "Claims withheld / rejected"
                  : latestEventForAgent(spec.agent_id) ||
                    specTask?.title ||
                    spec.specialty ||
                    "Executing task...";

                return (
                  <div key={spec.agent_id} className={`desk-card desk-specialist ${mood === "error" ? "desk-error" : ""}`}>
                    <div className="speech-bubble">{bubble.slice(0, 48)}</div>
                    <CharacterAvatar profile={profile} mood={mood} size={58} />
                    <div className="desk-name">{spec.role}</div>
                    <div className="desk-role">{allRejectedOrStale ? "withheld" : tStatus}</div>
                  </div>
                );
              })}

              {mode === "network" && (specialists.length > 0 || verdicts.length > 0) && (
                <div className={`desk-card desk-verifier ${stage.blocked ? "desk-error" : ""}`}>
                  <div className="speech-bubble">
                    {stage.blocked
                      ? "Blocked ungrounded release"
                      : verdicts.length > 0
                      ? `${verdicts.length} claim verdict(s) recorded`
                      : stage.step >= 3
                      ? "Checking claims & evidence..."
                      : "Standing by for review"}
                  </div>
                  <CharacterAvatar profile={verifierProfile} mood={verifierMood} size={58} />
                  <div className="desk-name">Reviewer</div>
                  <div className="desk-role">{stage.blocked ? "blocked" : "Verification"}</div>
                </div>
              )}
            </div>
          </div>

          <div className="office-stepper">
            {[
              { n: 1, label: "Team Formed" },
              { n: 2, label: "Executing Tasks" },
              { n: 3, label: "Reviewing Results" },
              { n: 4, label: stage.blocked ? "Release Blocked" : "Finalizing Answer" },
            ].map((s) => {
              const isErrStep = stage.blocked && s.n === 4;
              return (
                <div
                  key={s.n}
                  className={`step-pill ${
                    isErrStep
                      ? "step-error"
                      : stage.step > s.n || (stage.step === 4 && s.n === 4 && !stage.blocked)
                      ? "step-done"
                      : stage.step === s.n
                      ? "step-active"
                      : ""
                  }`}
                >
                  <span className="step-dot">{isErrStep ? "✕" : stage.step > s.n ? "✓" : s.n}</span>
                  <span>{s.label}</span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
};
