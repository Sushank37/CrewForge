import { describe, expect, it } from "vitest";
import fullRecordedTrace from "./fixtures/run_0edd437ada29.json";
import {
  OFFICE_DOOR,
  REVIEW_BAY,
  SPECIALIST_DESKS,
  applyTraceToOffice,
  classifyReleaseOutcome,
  createInitialOfficeState,
  stepOfficeSimulation,
} from "./officeEngine";
import type { TracePayload } from "../types";

describe("2D Office Animation Engine (complete recorded run_0edd437ada29 trace)", () => {
  it("replays all 48 persisted events, routes messages strictly by persisted IDs, and distinguishes partial release", () => {
    const trace = fullRecordedTrace as unknown as TracePayload;
    expect(trace.execution_events.length).toBe(48);

    const s0 = createInitialOfficeState();
    const s1 = applyTraceToOffice(s0, trace, false, 1000);

    // All 3 dynamic specialists spawn at entrance door targeting their desks
    expect(s1.actors.agt_7ada29_1_74d3.x).toBe(OFFICE_DOOR.x);
    expect(s1.actors.agt_7ada29_1_74d3.targetX).toBe(SPECIALIST_DESKS[0].x);
    expect(s1.actors.agt_7ada29_2_fe84.targetX).toBe(SPECIALIST_DESKS[1].x);
    expect(s1.actors.agt_7ada29_3_3a2c.targetX).toBe(SPECIALIST_DESKS[2].x);

    // Exactly 2 message packets corresponding to seq 29 & 30 (never invented from messages_delivered)
    const msgPackets = s1.packets.filter((p) => p.kind === "message");
    expect(msgPackets.length).toBe(2);
    for (const pkt of msgPackets) {
      expect(pkt.fromX).toBe(REVIEW_BAY.x);
      expect(pkt.toX).toBe(SPECIALIST_DESKS[2].x);
    }

    // run_0edd437ada29 is a Partial Verified Release (4 supported claims, 1 stale withheld), NOT blocked
    expect(classifyReleaseOutcome(trace)).toBe("partial");
    expect(s1.actors.lead.pose).toBe("completed");
    expect(s1.actors.reviewer.pose).toBe("reviewing");

    // Simulation steps specialists smoothly toward their desks
    const { next: s2, hasMotion } = stepOfficeSimulation(s1, 0.5, 1500);
    expect(hasMotion).toBe(true);
    expect(s2.actors.agt_7ada29_1_74d3.x).toBeGreaterThan(OFFICE_DOOR.x);
  });

  it("never invents a communication path when message_sent lacks an explicit recipient", () => {
    const s0 = createInitialOfficeState();
    const traceMissingRecipient: TracePayload = {
      run_id: "run_no_recip",
      mode: "network",
      status: "running",
      provider: "openai_compatible",
      model: "openai/gpt-oss-120b",
      is_historical_replay: false,
      roster: [{ agent_id: "agt_1", role: "Analyst" }],
      tasks: [],
      execution_events: [
        { seq: 1, event_type: "message_sent", actor: "agt_1", payload: {} },
        { seq: 2, event_type: "messages_delivered", actor: "agt_1", payload: {} },
      ],
      peer_messages: [],
      evidence: [],
      claims: [],
      review_verdicts: [],
      tool_executions: [],
    };
    const s1 = applyTraceToOffice(s0, traceMissingRecipient, true, 1000);
    expect(s1.packets.filter((p) => p.kind === "message").length).toBe(0);
  });

  it("distinguishes full, partial, and completely blocked release outcomes", () => {
    const base: TracePayload = {
      run_id: "run_outcomes",
      mode: "network",
      status: "completed",
      provider: "openai_compatible",
      model: "openai/gpt-oss-120b",
      is_historical_replay: false,
      roster: [],
      tasks: [],
      execution_events: [],
      peer_messages: [],
      evidence: [],
      claims: [],
      review_verdicts: [],
      tool_executions: [],
    };

    expect(
      classifyReleaseOutcome({
        ...base,
        honesty_gate: { passed: true, release_mode: "full" },
      })
    ).toBe("full");

    expect(
      classifyReleaseOutcome({
        ...base,
        status: "partial",
        honesty_gate: {
          passed: false,
          release_mode: "partial",
          verification_summary: { supported_claim_count: 3 },
        },
      })
    ).toBe("partial");

    expect(
      classifyReleaseOutcome({
        ...base,
        status: "partial",
        honesty_gate: {
          passed: false,
          release_mode: "blocked",
          verification_summary: { supported_claim_count: 0 },
        },
      })
    ).toBe("blocked");
  });

  it("switches characters to honest waiting pose during provider quota waits", () => {
    const s0 = createInitialOfficeState();
    const waitingTrace: TracePayload = {
      run_id: "run_wait",
      mode: "network",
      status: "running",
      provider: "openai_compatible",
      model: "openai/gpt-oss-120b",
      is_historical_replay: false,
      waiting_for_provider: true,
      provider_wait_seconds: 42.5,
      waiting_reason: "Waiting 42.5s for Groq quota reset.",
      roster: [],
      tasks: [],
      execution_events: [],
      peer_messages: [],
      evidence: [],
      claims: [],
      review_verdicts: [],
      tool_executions: [],
    };

    const s1 = applyTraceToOffice(s0, waitingTrace, true, 2000);
    expect(s1.actors.lead.pose).toBe("waiting");
    expect(s1.actors.lead.bubble).toContain("42.5s");
  });
});
