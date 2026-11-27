import {
  CHARACTER_ROSTER,
  CharacterProfile,
  pickSpecialistCharacter,
  pickVerifierCharacter,
} from "../characters/CharacterSet";
import type { SpritePose } from "../characters/FullBodyCharacter";
import type { TraceEvent, TracePayload } from "../types";

export const OFFICE_WIDTH = 960;
export const OFFICE_HEIGHT = 440;

export const OFFICE_DOOR = { x: 58, y: 215 };
export const LEAD_DESK = { x: 480, y: 158 };
export const REVIEW_BAY = { x: 820, y: 312 };
export const SPECIALIST_DESKS = [
  { x: 185, y: 318 },
  { x: 395, y: 318 },
  { x: 605, y: 318 },
  { x: 165, y: 215 },
];

export type OperationalState =
  | "idle"
  | "active_inference"
  | "dependency_waiting"
  | "quota_waiting"
  | "reviewing"
  | "revising"
  | "failed"
  | "completed";

export interface OfficeLayout {
  width: number;
  height: number;
  cols: number;
  rows: number;
  leadDesk: { x: number; y: number };
  reviewBay: { x: number; y: number };
  desks: Array<{ x: number; y: number; row: number; col: number }>;
}

export function computeOfficeLayout(specialistCount: number): OfficeLayout {
  if (specialistCount <= 3) {
    return {
      width: OFFICE_WIDTH,
      height: OFFICE_HEIGHT,
      cols: 3,
      rows: 1,
      leadDesk: LEAD_DESK,
      reviewBay: REVIEW_BAY,
      desks: SPECIALIST_DESKS.slice(0, 3).map((d, idx) => ({
        x: d.x,
        y: d.y,
        row: 0,
        col: idx,
      })),
    };
  }
  const cols = specialistCount <= 6 ? 3 : 4;
  const rows = Math.ceil(specialistCount / cols);
  const colSpacing = 215;
  const rowSpacing = 132;
  const leftMargin = 165;
  const startY = 305;
  const width = Math.max(OFFICE_WIDTH, leftMargin + (cols - 1) * colSpacing + 290);
  const height = Math.max(OFFICE_HEIGHT, startY + (rows - 1) * rowSpacing + 125);
  const desks: Array<{ x: number; y: number; row: number; col: number }> = [];
  for (let i = 0; i < specialistCount; i += 1) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    desks.push({
      x: leftMargin + col * colSpacing,
      y: startY + row * rowSpacing,
      row,
      col,
    });
  }
  return {
    width,
    height,
    cols,
    rows,
    leadDesk: { x: Math.round(width / 2), y: LEAD_DESK.y },
    reviewBay: { x: width - 130, y: 312 },
    desks,
  };
}

export interface OfficeActorState {
  id: string;
  role: string;
  profile: CharacterProfile;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  facing: "left" | "right";
  pose: SpritePose;
  arrivedPose: SpritePose;
  bubble: string | null;
  bubbleUntil: number;
  deskPos: { x: number; y: number };
  operationalState?: OperationalState;
  currentOperation?: string;
}

export type PacketKind = "task" | "message" | "result" | "revision" | "review";

export interface OfficePacket {
  id: string;
  kind: PacketKind;
  label: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  startTime: number;
  durationMs: number;
  color: string;
}

export interface OfficeSimState {
  runId: string | null;
  actors: Record<string, OfficeActorState>;
  packets: OfficePacket[];
  processedEventKeys: Set<string>;
  lastInteractionNote: string;
}

export function createInitialOfficeState(): OfficeSimState {
  return {
    runId: null,
    actors: {
      lead: {
        id: "lead",
        role: "Lead Orchestrator",
        profile: CHARACTER_ROSTER[0],
        x: LEAD_DESK.x,
        y: LEAD_DESK.y,
        targetX: LEAD_DESK.x,
        targetY: LEAD_DESK.y,
        facing: "right",
        pose: "idle",
        arrivedPose: "idle",
        bubble: null,
        bubbleUntil: 0,
        deskPos: LEAD_DESK,
      },
      reviewer: {
        id: "reviewer",
        role: "Verification Reviewer",
        profile: pickVerifierCharacter("reviewer_evidence"),
        x: REVIEW_BAY.x,
        y: REVIEW_BAY.y,
        targetX: REVIEW_BAY.x,
        targetY: REVIEW_BAY.y,
        facing: "left",
        pose: "idle",
        arrivedPose: "idle",
        bubble: null,
        bubbleUntil: 0,
        deskPos: REVIEW_BAY,
      },
    },
    packets: [],
    processedEventKeys: new Set<string>(),
    lastInteractionNote: "Office standing by.",
  };
}

export type ReleaseOutcome = "none" | "full" | "partial" | "blocked";

export function classifyReleaseOutcome(trace: TracePayload | null): ReleaseOutcome {
  if (!trace) return "none";
  if (String(trace.mode || "").toLowerCase() === "baseline") return "none";
  const status = String(trace.status || "").toLowerCase();
  if (status === "failed" || status === "cancelled") return "blocked";
  const gate = trace.honesty_gate;
  if (gate) {
    const mode = String(gate.release_mode || "").toLowerCase();
    const supported =
      gate.verification_summary?.supported_claim_count ??
      (trace.claims || []).filter((c) => String(c.status).toLowerCase() === "supported").length;
    if (mode === "blocked" || (gate.passed === false && mode !== "partial" && supported === 0)) {
      return "blocked";
    }
    if (mode === "partial" || status === "partial" || (gate.passed === false && supported > 0)) {
      return "partial";
    }
    if (gate.passed === true || mode === "full") {
      return "full";
    }
  }
  if (status === "partial") return "partial";
  if (status === "completed") return "full";
  return "none";
}

function eventKey(ev: TraceEvent, index: number): string {
  return ev.event_id || `${ev.seq ?? index}:${ev.event_type}:${ev.actor || ""}`;
}

function resolveRecipientFromEvent(ev: TraceEvent): string | null {
  const payload = ev.payload || {};
  const explicit =
    (payload.recipient_id as string | undefined) ||
    (payload.recipient_agent_id as string | undefined) ||
    (payload.to_agent_id as string | undefined) ||
    (payload.target_agent_id as string | undefined);
  if (explicit) return explicit;
  const msg = ev.message || "";
  const m = msg.match(/\bto\s+(agt_[A-Za-z0-9_]+|lead|reviewer_[A-Za-z0-9_]+)\b/i);
  return m ? m[1] : null;
}

function resolveActorPosStrict(
  actors: Record<string, OfficeActorState>,
  actorId: string | undefined | null
): { x: number; y: number } | null {
  if (!actorId) return null;
  if (actorId === "lead" || actorId === "orchestrator" || actorId === "honesty_gate") {
    return actors.lead?.deskPos || LEAD_DESK;
  }
  if (actorId.startsWith("reviewer")) {
    return actors.reviewer?.deskPos || REVIEW_BAY;
  }
  return actors[actorId]?.deskPos || null;
}

function findDependentDesk(
  trace: TracePayload,
  actors: Record<string, OfficeActorState>,
  completedActorId: string | undefined
): { x: number; y: number } {
  if (!completedActorId) return REVIEW_BAY;
  const tasks = trace.tasks || [];
  const completedTaskIds = new Set(
    tasks.filter((t) => t.assigned_agent_id === completedActorId).map((t) => t.task_id)
  );
  const dependent = tasks.find((t) => {
    const deps = t.depends_on_task_ids || t.depends_on || [];
    return deps.some((d) => completedTaskIds.has(d)) && t.assigned_agent_id !== completedActorId;
  });
  if (dependent && actors[dependent.assigned_agent_id]) {
    return actors[dependent.assigned_agent_id].deskPos;
  }
  return REVIEW_BAY;
}

export function applyTraceToOffice(
  prev: OfficeSimState,
  trace: TracePayload | null,
  isRunning: boolean,
  nowMs: number
): OfficeSimState {
  // Reset when switching runs or clearing chat
  if (!trace) {
    const fresh = createInitialOfficeState();
    if (isRunning) {
      fresh.actors.lead.pose = "working";
      fresh.actors.lead.arrivedPose = "working";
      fresh.actors.lead.bubble = "Analyzing prompt...";
      fresh.actors.lead.bubbleUntil = nowMs + 8000;
      fresh.lastInteractionNote = "Lead is analyzing the prompt and forming the team.";
    }
    return fresh;
  }

  const base =
    prev.runId && prev.runId !== trace.run_id ? createInitialOfficeState() : prev;
  const actors: Record<string, OfficeActorState> = { ...base.actors };
  const packets = base.packets.filter((p) => p.startTime + p.durationMs > nowMs);
  const processed = new Set(base.processedEventKeys);
  let note = base.lastInteractionNote;

  // Ensure every dynamic specialist from roster gets a unique, non-overlapping workstation
  const specialists = (trace.roster || []).filter((a) => a.role.toLowerCase() !== "lead");
  const layout = computeOfficeLayout(specialists.length);
  actors.lead = {
    ...actors.lead,
    deskPos: layout.leadDesk,
    targetX: layout.leadDesk.x,
    targetY: layout.leadDesk.y,
    x: prev.runId === trace.run_id ? actors.lead.x : layout.leadDesk.x,
    y: prev.runId === trace.run_id ? actors.lead.y : layout.leadDesk.y,
  };
  actors.reviewer = {
    ...actors.reviewer,
    deskPos: layout.reviewBay,
    targetX: layout.reviewBay.x,
    targetY: layout.reviewBay.y,
    x: prev.runId === trace.run_id ? actors.reviewer.x : layout.reviewBay.x,
    y: prev.runId === trace.run_id ? actors.reviewer.y : layout.reviewBay.y,
  };
  specialists.forEach((spec, idx) => {
    const desk = layout.desks[idx];
    if (!actors[spec.agent_id]) {
      actors[spec.agent_id] = {
        id: spec.agent_id,
        role: spec.role,
        profile: pickSpecialistCharacter(spec.role, idx),
        x: OFFICE_DOOR.x,
        y: OFFICE_DOOR.y,
        targetX: desk.x,
        targetY: desk.y,
        facing: desk.x >= OFFICE_DOOR.x ? "right" : "left",
        pose: "walking",
        arrivedPose: "idle",
        bubble: "Joining team",
        bubbleUntil: nowMs + 3500,
        deskPos: desk,
        operationalState: "idle",
        currentOperation: "Joining team",
      };
    } else if (
      actors[spec.agent_id].deskPos.x !== desk.x ||
      actors[spec.agent_id].deskPos.y !== desk.y
    ) {
      actors[spec.agent_id] = {
        ...actors[spec.agent_id],
        deskPos: desk,
        targetX: desk.x,
        targetY: desk.y,
      };
    }
  });

  // Update reviewer identity from latest verdict if available
  const verdicts = trace.review_verdicts || [];
  if (verdicts.length > 0) {
    const lastRev = verdicts[verdicts.length - 1].reviewer_id;
    actors.reviewer = {
      ...actors.reviewer,
      profile: pickVerifierCharacter(lastRev),
    };
  }

  // Process newly arrived batch of execution events in exact sequence
  const events = trace.execution_events || [];
  let stagger = 0;

  events.forEach((ev, idx) => {
    const key = eventKey(ev, idx);
    if (processed.has(key)) return;
    processed.add(key);

    const tStart = nowMs + stagger * 260;
    const actorId = ev.actor || "lead";
    const actor = actors[actorId];

    switch (ev.event_type) {
      case "run_started":
      case "plan_created": {
        actors.lead = {
          ...actors.lead,
          pose: "working",
          arrivedPose: "working",
          bubble: ev.event_type === "plan_created" ? "Team assembled!" : "Planning DAG...",
          bubbleUntil: tStart + 3200,
        };
        note = ev.message || "Lead assembled dynamic specialist team.";
        break;
      }
      case "task_started": {
        if (actor) {
          actors[actorId] = {
            ...actor,
            arrivedPose: "working",
            pose: Math.hypot(actor.x - actor.targetX, actor.y - actor.targetY) > 6 ? "walking" : "working",
            bubble: "Assigned task",
            bubbleUntil: tStart + 2800,
          };
          packets.push({
            id: `pkt_${key}`,
            kind: "task",
            label: "Task",
            fromX: LEAD_DESK.x,
            fromY: LEAD_DESK.y - 8,
            toX: actor.deskPos.x,
            toY: actor.deskPos.y - 8,
            startTime: tStart,
            durationMs: 950,
            color: "#6366f1",
          });
          stagger += 1;
          note = ev.message || `Assigned task to ${actor.role}.`;
        }
        break;
      }
      case "tool_executed":
      case "claims_recorded": {
        if (actor) {
          actors[actorId] = {
            ...actor,
            pose: "working",
            arrivedPose: "working",
            bubble: ev.event_type === "tool_executed" ? "Computing tool" : "Recording claim",
            bubbleUntil: tStart + 2400,
          };
          note = ev.message || `${actor.role} executed deterministic check.`;
        }
        break;
      }
      case "task_completed": {
        if (actor) {
          const dest = findDependentDesk(trace, actors, actorId);
          actors[actorId] = {
            ...actor,
            pose: "completed",
            arrivedPose: "completed",
            bubble: "Task done ✓",
            bubbleUntil: tStart + 2600,
          };
          packets.push({
            id: `pkt_${key}`,
            kind: "result",
            label: "Output",
            fromX: actor.deskPos.x,
            fromY: actor.deskPos.y - 10,
            toX: dest.x,
            toY: dest.y - 10,
            startTime: tStart,
            durationMs: 1000,
            color: "#10b981",
          });
          stagger += 1;
          note = ev.message || `${actor.role} delivered completed output.`;
        }
        break;
      }
      case "message_sent": {
        const recipient = resolveRecipientFromEvent(ev);
        const fromPos = resolveActorPosStrict(actors, actorId);
        const toPos = resolveActorPosStrict(actors, recipient);
        if (fromPos && toPos && (fromPos.x !== toPos.x || fromPos.y !== toPos.y)) {
          packets.push({
            id: `pkt_${key}`,
            kind: "message",
            label: "Msg",
            fromX: fromPos.x,
            fromY: fromPos.y - 14,
            toX: toPos.x,
            toY: toPos.y - 14,
            startTime: tStart,
            durationMs: 1050,
            color: "#38bdf8",
          });
          stagger += 1;
        }
        note = ev.message || "Agent collaboration message sent.";
        break;
      }
      case "messages_delivered": {
        if (actor) {
          actors[actorId] = {
            ...actor,
            bubble: "Received feedback",
            bubbleUntil: tStart + 2400,
          };
        }
        note = ev.message || "Delivered inbound collaboration message.";
        break;
      }
      case "review_completed":
      case "budget_reserved": {
        if (actorId.startsWith("reviewer")) {
          actors.reviewer = {
            ...actors.reviewer,
            pose: "reviewing",
            arrivedPose: "reviewing",
            bubble: ev.event_type === "review_completed" ? "Review complete" : "Inspecting claims...",
            bubbleUntil: tStart + 3000,
          };
          if (ev.event_type === "review_completed") {
            packets.push({
              id: `pkt_${key}`,
              kind: "review",
              label: "Verdict",
              fromX: REVIEW_BAY.x,
              fromY: REVIEW_BAY.y - 10,
              toX: LEAD_DESK.x,
              toY: LEAD_DESK.y - 10,
              startTime: tStart,
              durationMs: 950,
              color: "#f59e0b",
            });
            stagger += 1;
          }
          note = ev.message || "Reviewer inspected submitted claims.";
        }
        break;
      }
      case "claim_revised":
      case "claims_invalidated": {
        if (actor) {
          actors[actorId] = {
            ...actor,
            pose: "working",
            arrivedPose: "working",
            bubble: "Revising claim ↩",
            bubbleUntil: tStart + 3000,
          };
          packets.push({
            id: `pkt_${key}`,
            kind: "revision",
            label: "Revise",
            fromX: REVIEW_BAY.x,
            fromY: REVIEW_BAY.y - 12,
            toX: actor.deskPos.x,
            toY: actor.deskPos.y - 12,
            startTime: tStart,
            durationMs: 1050,
            color: "#fb7185",
          });
          stagger += 1;
          note = ev.message || `${actor.role} revising claim after review feedback.`;
        }
        break;
      }
      case "honesty_gate_evaluated":
      case "synthesis_fast_path":
      case "synthesis_completed":
      case "run_completed": {
        if (String(trace.mode || "").toLowerCase() === "baseline") {
          actors.lead = {
            ...actors.lead,
            pose: "idle",
            arrivedPose: "idle",
            bubble: "Baseline (unverified)",
            bubbleUntil: tStart + 5000,
          };
          actors.reviewer = {
            ...actors.reviewer,
            pose: "idle",
            arrivedPose: "idle",
            bubble: "No review (baseline)",
            bubbleUntil: tStart + 5000,
          };
          note = ev.message || "Baseline completed without independent verification.";
          break;
        }
        const outcome = classifyReleaseOutcome(trace);
        const leadPose: SpritePose = outcome === "blocked" ? "error" : "completed";
        const revPose: SpritePose =
          outcome === "blocked" ? "error" : outcome === "partial" ? "reviewing" : "completed";
        actors.lead = {
          ...actors.lead,
          pose: leadPose,
          arrivedPose: leadPose,
          bubble:
            outcome === "blocked"
              ? "Release blocked"
              : outcome === "partial"
              ? "Partial verified ✓"
              : "Answer ready ✓",
          bubbleUntil: tStart + 5000,
        };
        actors.reviewer = {
          ...actors.reviewer,
          pose: revPose,
          arrivedPose: revPose,
          bubble:
            outcome === "blocked"
              ? "Gate withheld"
              : outcome === "partial"
              ? "Withheld unverified"
              : "Verified ✓",
          bubbleUntil: tStart + 5000,
        };
        note =
          ev.message ||
          (outcome === "blocked"
            ? "Honesty Gate blocked unsafe release."
            : outcome === "partial"
            ? "Released verified subset; unverified claims withheld."
            : "Verified answer delivered.");
        break;
      }
      default:
        break;
    }
  });

  // Derive honest per-actor operationalState and currentOperation from trace state
  const tasks = trace.tasks || [];
  const completedTaskIds = new Set(
    tasks.filter((t) => String(t.status).toLowerCase() === "completed").map((t) => t.task_id)
  );
  Object.keys(actors).forEach((id) => {
    const actor = actors[id];
    if (id === "lead") {
      if (isRunning) {
        actor.operationalState = trace.waiting_for_provider
          ? "quota_waiting"
          : events.some((e) => e.event_type === "synthesis_started")
            ? "active_inference"
            : "active_inference";
        actor.currentOperation = trace.waiting_for_provider
          ? trace.waiting_reason || "Waiting for provider quota"
          : events.some((e) => e.event_type === "synthesis_started")
            ? "Synthesizing verified final answer"
            : "Orchestrating execution plan";
      } else {
        const outcome = classifyReleaseOutcome(trace);
        actor.operationalState =
          trace.status === "failed" || outcome === "blocked" ? "failed" : "completed";
        actor.currentOperation =
          outcome === "blocked"
            ? "Release blocked by Honesty Gate"
            : outcome === "partial"
              ? "Partial verified release"
              : "Delivered final answer";
      }
      return;
    }
    if (id === "reviewer") {
      if (isRunning && trace.waiting_for_provider) {
        actor.operationalState = "quota_waiting";
        actor.currentOperation = "Paused during provider quota wait";
      } else if (actor.pose === "reviewing") {
        actor.operationalState = "reviewing";
        actor.currentOperation = `Inspecting ${verdicts.length} verdict(s)`;
      } else if (!isRunning) {
        actor.operationalState = trace.status === "failed" ? "failed" : "completed";
        actor.currentOperation = `${verdicts.length} review verdict(s) recorded`;
      } else {
        actor.operationalState = "idle";
        actor.currentOperation = "Standing by for specialist claims";
      }
      return;
    }
    const myTasks = tasks.filter((t) => t.assigned_agent_id === id);
    const activeTask = myTasks.find((t) => String(t.status).toLowerCase() === "running" || String(t.status).toLowerCase() === "in_progress");
    const failedTask = myTasks.find((t) => String(t.status).toLowerCase() === "failed");
    const pendingTask = myTasks.find((t) => String(t.status).toLowerCase() === "pending");
    if (isRunning && trace.waiting_for_provider) {
      actor.operationalState = "quota_waiting";
      actor.currentOperation = trace.waiting_reason || "Waiting for provider quota";
    } else if (failedTask || trace.status === "failed") {
      actor.operationalState = "failed";
      actor.currentOperation = failedTask ? `Failed: ${failedTask.title || failedTask.task_id}` : "Run failed";
    } else if (activeTask) {
      actor.operationalState = "active_inference";
      actor.currentOperation = `Working: ${activeTask.title || activeTask.task_id}`;
    } else if (pendingTask) {
      const deps = pendingTask.depends_on_task_ids || pendingTask.depends_on || [];
      const waitingDeps = deps.filter((d) => !completedTaskIds.has(d));
      if (waitingDeps.length > 0) {
        actor.operationalState = "dependency_waiting";
        actor.currentOperation = `Waiting on ${waitingDeps.join(", ")}`;
      } else {
        actor.operationalState = "idle";
        actor.currentOperation = `Queued: ${pendingTask.title || pendingTask.task_id}`;
      }
    } else if (myTasks.length > 0 && myTasks.every((t) => String(t.status).toLowerCase() === "completed")) {
      actor.operationalState = "completed";
      actor.currentOperation = `Completed ${myTasks.length} task(s)`;
    } else if (actor.pose === "working") {
      actor.operationalState = "active_inference";
      actor.currentOperation = actor.bubble || "Running inference";
    }
  });

  // Honest provider-quota waiting state overrides active typing poses while paused
  if (isRunning && trace.waiting_for_provider) {
    const waitText = trace.provider_wait_seconds
      ? `Quota wait (${trace.provider_wait_seconds}s)`
      : "Waiting for quota...";
    Object.keys(actors).forEach((id) => {
      if (actors[id].pose !== "walking") {
        actors[id] = {
          ...actors[id],
          pose: "waiting",
          bubble: id === "lead" ? waitText : null,
          bubbleUntil: nowMs + 4000,
        };
      }
    });
    note = trace.waiting_reason || waitText;
  } else if (!isRunning && (trace.status === "completed" || trace.status === "partial" || trace.status === "failed")) {
    if (String(trace.mode || "").toLowerCase() === "baseline") {
      actors.lead = { ...actors.lead, pose: "idle", arrivedPose: "idle" };
      actors.reviewer = { ...actors.reviewer, pose: "idle", arrivedPose: "idle" };
      return {
        runId: trace.run_id,
        actors,
        packets,
        processedEventKeys: processed,
        lastInteractionNote: "Baseline completed without independent verification.",
      };
    }
    const outcome = classifyReleaseOutcome(trace);
    const finalLeadPose: SpritePose = outcome === "blocked" ? "error" : "completed";
    const finalRevPose: SpritePose =
      outcome === "blocked" ? "error" : outcome === "partial" ? "reviewing" : "completed";
    actors.lead = { ...actors.lead, pose: finalLeadPose, arrivedPose: finalLeadPose };
    actors.reviewer = { ...actors.reviewer, pose: finalRevPose, arrivedPose: finalRevPose };
    Object.keys(actors).forEach((id) => {
      if (id !== "lead" && id !== "reviewer" && actors[id].pose === "waiting") {
        actors[id] = { ...actors[id], pose: "completed" };
      }
    });
  }

  return {
    runId: trace.run_id,
    actors,
    packets,
    processedEventKeys: processed,
    lastInteractionNote: note,
  };
}

export function stepOfficeSimulation(
  state: OfficeSimState,
  dtSeconds: number,
  nowMs: number
): { next: OfficeSimState; hasMotion: boolean } {
  const speed = 175; // px/s walking speed
  let hasMotion = false;
  const nextActors: Record<string, OfficeActorState> = {};

  for (const [id, actor] of Object.entries(state.actors)) {
    const dx = actor.targetX - actor.x;
    const dy = actor.targetY - actor.y;
    const dist = Math.hypot(dx, dy);
    let x = actor.x;
    let y = actor.y;
    let pose = actor.pose;
    let facing = actor.facing;

    if (dist > 2) {
      hasMotion = true;
      const step = Math.min(dist, speed * dtSeconds);
      x += (dx / dist) * step;
      y += (dy / dist) * step;
      pose = "walking";
      if (Math.abs(dx) > 1) facing = dx >= 0 ? "right" : "left";
    } else if (pose === "walking") {
      x = actor.targetX;
      y = actor.targetY;
      pose = actor.arrivedPose;
    }

    const bubble = actor.bubble && nowMs < actor.bubbleUntil ? actor.bubble : null;
    if (bubble) hasMotion = true;

    nextActors[id] = { ...actor, x, y, pose, facing, bubble };
  }

  const activePackets = state.packets.filter((p) => p.startTime + p.durationMs > nowMs);
  if (activePackets.length > 0) hasMotion = true;

  return {
    next: {
      ...state,
      actors: nextActors,
      packets: activePackets,
    },
    hasMotion,
  };
}
