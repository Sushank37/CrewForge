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

function deriveStage(trace: TracePayload | null, isRunning: boolean): {
  step: number; // 0 idle, 1 team formed, 2 executing, 3 reviewing, 4 finalized
  headline: string;
  subline: string;
  leadBubble: string;
} {
  if (!trace) {
    if (isRunning) {
      return {
        step: 1,
        headline: "Your AI Team is assembling...",
        subline: "Lead is analyzing the prompt and forming the team.",
        leadBubble: "Analyzing prompt & planning...",
      };
    }
    return {
      step: 0,
      headline: "CrewForge AI Office",
      subline: "Ask a question below to watch your dynamic team collaborate in real time.",
      leadBubble: "Ready when you are.",
    };
  }

  const events = trace.execution_events || [];
  const types = new Set(events.map((e) => e.event_type));
  const status = String(trace.status || "").toLowerCase();

  if (status === "completed" || status === "partial" || types.has("run_completed")) {
    return {
      step: 4,
      headline: status === "completed" ? "Mission Accomplished" : "Completed with Verified Safeguards",
      subline: `${trace.roster?.length || 1} agents collaborated · ${events.length} persisted events`,
      leadBubble: "Verified & delivered final answer!",
    };
  }
  if (status === "failed" || types.has("run_failed")) {
    return {
      step: 4,
      headline: "Execution Halted",
      subline: "Review the error or Mission Control trace for details.",
      leadBubble: "Run encountered an error.",
    };
  }
  if (types.has("honesty_gate_evaluated") || types.has("synthesis_completed")) {
    return {
      step: 4,
      headline: "Finalizing Verified Answer...",
      subline: "Honesty Gate is checking claim provenance and release rules.",
      leadBubble: "Running Honesty Gate...",
    };
  }
  if (types.has("review_completed") || types.has("claim_revised")) {
    return {
      step: 3,
      headline: "Reviewing Claims & Evidence...",
      subline: "Independent reviewers are verifying arithmetic and provenance.",
      leadBubble: "Checking reviewer verdicts...",
    };
  }
  if (types.has("plan_created") || types.has("task_started")) {
    return {
      step: 2,
      headline: "Your AI Team is on it!",
      subline: "Lead has assembled the team and specialists are executing tasks.",
      leadBubble: "Coordinating specialist tasks...",
    };
  }
  return {
    step: 1,
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
  const verdicts = trace?.review_verdicts || [];
  const events = trace?.execution_events || [];

  const elapsed =
    trace?.latency_ms !== null && trace?.latency_ms !== undefined
      ? `${(trace.latency_ms / 1000).toFixed(1)}s`
      : isRunning
      ? "Live"
      : "—";

  const leadMood: CharacterMood =
    stage.step === 4
      ? trace?.status === "failed"
        ? "error"
        : "completed"
      : isRunning || stage.step > 0
      ? "working"
      : "idle";

  const lastReviewerId =
    verdicts.length > 0 ? verdicts[verdicts.length - 1].reviewer_id : "reviewer_evidence";
  const verifierProfile = pickVerifierCharacter(lastReviewerId);
  const verifierMood: CharacterMood =
    verdicts.length > 0
      ? stage.step === 4
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
          <span className={`office-pill ${isRunning ? "pill-live" : "pill-idle"}`}>
            <span className="pulse-dot" />
            {isRunning ? "Live Execution" : trace ? String(trace.status).toUpperCase() : mode === "baseline" ? "Baseline" : "Standby"}
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
            {/* Whiteboard */}
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
                <span className={stage.step >= 4 ? "wb-done" : stage.step === 3 ? "wb-active" : "wb-todo"}>
                  {stage.step >= 4 ? "✓" : "○"} Review
                </span>
              </div>
              <div className="wb-row">
                <span className={stage.step >= 4 ? "wb-done" : "wb-todo"}>
                  {stage.step >= 4 ? "✓" : "○"} Finalize
                </span>
              </div>
            </div>

            {/* Lead Desk */}
            <div className="office-lead-row">
              <div className="desk-card desk-lead">
                <div className="speech-bubble">{stage.leadBubble}</div>
                <CharacterAvatar profile={leadProfile} mood={leadMood} size={68} />
                <div className="desk-name">Lead</div>
                <div className="desk-role">Orchestrator</div>
              </div>
            </div>

            {/* Animated connection bar */}
            {specialists.length > 0 && (
              <svg className="office-wires" viewBox="0 0 600 32" preserveAspectRatio="none" aria-hidden="true">
                <path d="M300 0 L300 14 M110 14 L490 14 M110 14 L110 32 M300 14 L300 32 M490 14 L490 32"
                  stroke="#60a5fa" strokeWidth="2" fill="none" strokeDasharray={isRunning ? "6 4" : undefined} />
              </svg>
            )}

            {/* Dynamic Specialist Desks + Verification Desk */}
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
                const mood: CharacterMood =
                  tStatus === "completed"
                    ? "completed"
                    : tStatus === "failed"
                    ? "error"
                    : tStatus === "running" || tStatus === "in_progress"
                    ? "working"
                    : isRunning
                    ? "working"
                    : "idle";
                const bubble =
                  latestEventForAgent(spec.agent_id) ||
                  specTask?.title ||
                  spec.specialty ||
                  "Executing task...";

                return (
                  <div key={spec.agent_id} className="desk-card desk-specialist">
                    <div className="speech-bubble">{bubble.slice(0, 48)}</div>
                    <CharacterAvatar profile={profile} mood={mood} size={58} />
                    <div className="desk-name">{spec.role}</div>
                    <div className="desk-role">{tStatus}</div>
                  </div>
                );
              })}

              {mode === "network" && (specialists.length > 0 || verdicts.length > 0) && (
                <div className="desk-card desk-verifier">
                  <div className="speech-bubble">
                    {verdicts.length > 0
                      ? `${verdicts.length} claim verdict(s) recorded`
                      : stage.step >= 3
                      ? "Checking claims & evidence..."
                      : "Standing by for review"}
                  </div>
                  <CharacterAvatar profile={verifierProfile} mood={verifierMood} size={58} />
                  <div className="desk-name">Reviewer</div>
                  <div className="desk-role">Verification</div>
                </div>
              )}
            </div>
          </div>

          {/* 4-Step Progress Stepper */}
          <div className="office-stepper">
            {[
              { n: 1, label: "Team Formed" },
              { n: 2, label: "Executing Tasks" },
              { n: 3, label: "Reviewing Results" },
              { n: 4, label: "Finalizing Answer" },
            ].map((s) => (
              <div
                key={s.n}
                className={`step-pill ${stage.step > s.n ? "step-done" : stage.step === s.n ? "step-active" : ""}`}
              >
                <span className="step-dot">{stage.step > s.n ? "✓" : s.n}</span>
                <span>{s.label}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
};
