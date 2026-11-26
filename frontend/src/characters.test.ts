import { describe, expect, it } from "vitest";
import {
  CHARACTER_ROSTER,
  pickSpecialistCharacter,
  pickVerifierCharacter,
} from "./characters/CharacterSet";

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
});
