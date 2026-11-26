import React, { useState } from "react";
import { CHARACTER_ROSTER, CharacterAvatar, CharacterMood } from "../characters/CharacterSet";

export const CharacterGallery: React.FC = () => {
  const [mood, setMood] = useState<CharacterMood>("idle");

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1>CrewForge Character Set (16)</h1>
          <p className="muted">
            1 Lead Orchestrator, 3 Verification Characters, and 12 Reusable Specialist Visual Identities.
            Specialist characters map dynamically to whatever roles the Lead Planner creates.
          </p>
        </div>
        <div className="mood-picker">
          {(["idle", "working", "reviewing", "completed", "error"] as CharacterMood[]).map((m) => (
            <button
              key={m}
              type="button"
              className={`btn-sm ${mood === m ? "btn-primary" : "btn-ghost"}`}
              onClick={() => setMood(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="character-grid">
        {CHARACTER_ROSTER.map((char) => (
          <div key={char.id} className="char-card">
            <CharacterAvatar profile={char} mood={mood} size={72} />
            <div className="char-name">{char.name}</div>
            <span className={`pill pill-${char.category}`}>{char.category}</span>
            <div className="char-badge">{char.defaultBadge}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
