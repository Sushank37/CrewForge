import { describe, expect, it } from "vitest";
import {
  CHARACTER_ROSTER,
  pickSpecialistCharacter,
  pickVerifierCharacter,
} from "./characters/CharacterSet";
import { deriveStage, isVerificationBlocked } from "./components/AiOfficeWorkspace";
import type { TracePayload } from "./types";

describe("CrewForge 16-Character Roster", () => {
  it("contains exactly 16 unique characters (1 lead, 3 verifiers, 12 specialists)", () => {
    expect(CHARACTER_ROSTER).toHaveLength(16);
    const ids = new Set(CHARACTER_ROSTER.map((c) => c.id));
    expect(ids.size).toBe(16);

    expect(CHARACTER_ROSTER.filter((c) => c.category === "lead")).toHaveLength(1);
    expect(CHARACTER_ROSTER.filter((c) => c.category === "verifier")).toHaveLength(3);
    expect(CHARACTER_ROSTER.filter((c) => c.category === "specialist")).toHaveLength(12);
  });

  it("maps arbitrary dynamic specialist roles deterministically to the 12-character pool", () => {
    const a = pickSpecialistCharacter("agt_custom_supply_chain", 0);
    const b = pickSpecialistCharacter("agt_custom_supply_chain", 0);
    expect(a.id).toBe(b.id);
    expect(a.category).toBe("specialist");

    expect(pickVerifierCharacter("reviewer_numbers").id).toBe("verifier_numbers");
    expect(pickVerifierCharacter("reviewer_consistency").id).toBe("verifier_consistency");
    expect(pickVerifierCharacter("reviewer_evidence").id).toBe("verifier_evidence");
  });

  it("never marks failed or Honesty-Gate-blocked runs as Mission Accomplished", () => {
    const blockedTrace: TracePayload = {
      run_id: "run_blocked",
      mode: "network",
      status: "partial",
      provider: "fake",
      model: "fake-v1",
      is_historical_replay: false,
      roster: [],
      tasks: [],
      execution_events: [{ seq: 1, event_type: "run_completed" }],
      peer_messages: [],
      evidence: [],
      claims: [],
      review_verdicts: [],
      tool_executions: [],
      honesty_gate: { passed: false, release_mode: "blocked" },
    };
    expect(isVerificationBlocked(blockedTrace)).toBe(true);
    const stage = deriveStage(blockedTrace, false);
    expect(stage.blocked).toBe(true);
    expect(stage.headline).toBe("Verification Blocked Release");
  });
});
