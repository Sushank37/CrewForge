import { describe, expect, it } from "vitest";
import {
  OFFICE_DOOR,
  SPECIALIST_DESKS,
  applyTraceToOffice,
  createInitialOfficeState,
  stepOfficeSimulation,
} from "./officeEngine";
import type { TracePayload } from "../types";

describe("2D Office Animation Engine (run_0edd437ada29 lifecycle)", () => {
  it("spawns specialists at the door, batches handovers/messages/revisions, and walks smoothly", () => {
    const trace: TracePayload = {
      run_id: "run_0edd437ada29",
      mode: "network",
      status: "running",
      provider: "openai_compatible",
      model: "openai/gpt-oss-120b",
      is_historical_replay: false,
      roster: [
        { agent_id: "agt_7ada29_1_74d3", role: "Cloud Cost Analyst" },
        { agent_id: "agt_7ada29_2_fe84", role: "Staffing Cost Analyst" },
        { agent_id: "agt_7ada29_3_3a2c", role: "Budget Aggregator" },
      ],
      tasks: [
        {
          task_id: "tsk_1",
          assigned_agent_id: "agt_7ada29_1_74d3",
          status: "completed",
        },
        {
          task_id: "tsk_2",
          assigned_agent_id: "agt_7ada29_2_fe84",
          status: "completed",
        },
        {
          task_id: "tsk_3",
          assigned_agent_id: "agt_7ada29_3_3a2c",
          depends_on_task_ids: ["tsk_1", "tsk_2"],
          status: "running",
        },
      ],
      execution_events: [
        { seq: 1, event_type: "run_started", actor: "lead", message: "Started" },
        { seq: 4, event_type: "plan_created", actor: "lead", message: "3 specialists" },
        { seq: 5, event_type: "task_started", actor: "agt_7ada29_1_74d3" },
        { seq: 6, event_type: "task_started", actor: "agt_7ada29_2_fe84" },
        { seq: 12, event_type: "task_completed", actor: "agt_7ada29_2_fe84" },
        { seq: 29, event_type: "message_sent", actor: "reviewer_combined", message: "Sent clarification message to agt_7ada29_3_3a2c." },
        { seq: 37, event_type: "claim_revised", actor: "agt_7ada29_3_3a2c" },
      ],
      peer_messages: [],
      evidence: [],
      claims: [],
      review_verdicts: [],
      tool_executions: [],
    };

    const s0 = createInitialOfficeState();
    const s1 = applyTraceToOffice(s0, trace, true, 1000, false);

    // All 3 specialists spawn at OFFICE_DOOR and target their respective desks
    expect(s1.actors.agt_7ada29_1_74d3.x).toBe(OFFICE_DOOR.x);
    expect(s1.actors.agt_7ada29_1_74d3.targetX).toBe(SPECIALIST_DESKS[0].x);
    expect(s1.actors.agt_7ada29_2_fe84.targetX).toBe(SPECIALIST_DESKS[1].x);
    expect(s1.actors.agt_7ada29_3_3a2c.targetX).toBe(SPECIALIST_DESKS[2].x);

    // Batch events generated task, result, message, and revision packets without dropping any
    const kinds = new Set(s1.packets.map((p) => p.kind));
    expect(kinds.has("task")).toBe(true);
    expect(kinds.has("result")).toBe(true);
    expect(kinds.has("message")).toBe(true);
    expect(kinds.has("revision")).toBe(true);

    // Stepping simulation moves specialists toward their desks
    const { next: s2, hasMotion } = stepOfficeSimulation(s1, 0.5, 1500);
    expect(hasMotion).toBe(true);
    expect(s2.actors.agt_7ada29_1_74d3.x).toBeGreaterThan(OFFICE_DOOR.x);
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

    const s1 = applyTraceToOffice(s0, waitingTrace, true, 2000, false);
    expect(s1.actors.lead.pose).toBe("waiting");
    expect(s1.actors.lead.bubble).toContain("42.5s");
  });
});
