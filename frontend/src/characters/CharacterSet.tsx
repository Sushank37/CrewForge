import React from "react";

export type CharacterCategory = "lead" | "verifier" | "specialist";
export type CharacterMood = "idle" | "working" | "reviewing" | "completed" | "error";

export interface CharacterProfile {
  id: string;
  name: string;
  category: CharacterCategory;
  defaultBadge: string;
  accent: string;
  hair: string;
  skin: string;
  outfit: string;
  accessory:
    | "headset"
    | "visor"
    | "glasses"
    | "beret"
    | "cap"
    | "goggles"
    | "headphones"
    | "scarf"
    | "beanie"
    | "none";
  hairStyle: "wavy" | "bob" | "spiky" | "bun" | "ponytail" | "curls" | "short" | "long";
}

export const CHARACTER_ROSTER: CharacterProfile[] = [
  // 1 Lead Orchestrator
  {
    id: "lead",
    name: "Lead Orchestrator",
    category: "lead",
    defaultBadge: "Command",
    accent: "#6366f1",
    hair: "#7c3f1d",
    skin: "#f6c7a6",
    outfit: "#3730a3",
    accessory: "headset",
    hairStyle: "wavy",
  },
  // 3 Verification Characters
  {
    id: "verifier_evidence",
    name: "Evidence Reviewer",
    category: "verifier",
    defaultBadge: "Provenance",
    accent: "#10b981",
    hair: "#1e293b",
    skin: "#f3c19d",
    outfit: "#065f46",
    accessory: "glasses",
    hairStyle: "bob",
  },
  {
    id: "verifier_numbers",
    name: "Numerical Reviewer",
    category: "verifier",
    defaultBadge: "Arithmetic",
    accent: "#06b6d4",
    hair: "#92400e",
    skin: "#e5a982",
    outfit: "#155e75",
    accessory: "visor",
    hairStyle: "short",
  },
  {
    id: "verifier_consistency",
    name: "Consistency Reviewer",
    category: "verifier",
    defaultBadge: "Logic",
    accent: "#f59e0b",
    hair: "#4c1d95",
    skin: "#f7d1b5",
    outfit: "#78350f",
    accessory: "beret",
    hairStyle: "bun",
  },
  // 12 Reusable Specialist Visual Identities
  {
    id: "spec_01",
    name: "Aria (Teal)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#14b8a6",
    hair: "#4c1d95",
    skin: "#f5c6a5",
    outfit: "#0f766e",
    accessory: "headphones",
    hairStyle: "bob",
  },
  {
    id: "spec_02",
    name: "Milo (Azure)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#3b82f6",
    hair: "#3b2314",
    skin: "#e6aa82",
    outfit: "#1d4ed8",
    accessory: "cap",
    hairStyle: "short",
  },
  {
    id: "spec_03",
    name: "Kai (Amber)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#f97316",
    hair: "#7c2d12",
    skin: "#f4c29f",
    outfit: "#9a3412",
    accessory: "goggles",
    hairStyle: "spiky",
  },
  {
    id: "spec_04",
    name: "Nora (Gold)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#eab308",
    hair: "#ca8a04",
    skin: "#f8d4ba",
    outfit: "#854d0e",
    accessory: "headphones",
    hairStyle: "long",
  },
  {
    id: "spec_05",
    name: "Soren (Silver)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#94a3b8",
    hair: "#cbd5e1",
    skin: "#eec1a2",
    outfit: "#334155",
    accessory: "glasses",
    hairStyle: "wavy",
  },
  {
    id: "spec_06",
    name: "Lina (Rose)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#ec4899",
    hair: "#831843",
    skin: "#f7ceb0",
    outfit: "#9d174d",
    accessory: "glasses",
    hairStyle: "ponytail",
  },
  {
    id: "spec_07",
    name: "Toby (Mint)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#22c55e",
    hair: "#1e293b",
    skin: "#c6865c",
    outfit: "#166534",
    accessory: "none",
    hairStyle: "curls",
  },
  {
    id: "spec_08",
    name: "Iris (Indigo)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#818cf8",
    hair: "#312e81",
    skin: "#f5c7a7",
    outfit: "#4338ca",
    accessory: "scarf",
    hairStyle: "bun",
  },
  {
    id: "spec_09",
    name: "Rex (Copper)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#fb923c",
    hair: "#451a03",
    skin: "#dca078",
    outfit: "#7c2d12",
    accessory: "beanie",
    hairStyle: "short",
  },
  {
    id: "spec_10",
    name: "Jade (Emerald)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#34d399",
    hair: "#065f46",
    skin: "#f2bfa0",
    outfit: "#047857",
    accessory: "headphones",
    hairStyle: "ponytail",
  },
  {
    id: "spec_11",
    name: "Devon (Slate)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#38bdf8",
    hair: "#0f172a",
    skin: "#b5774e",
    outfit: "#1e293b",
    accessory: "glasses",
    hairStyle: "short",
  },
  {
    id: "spec_12",
    name: "Zara (Violet)",
    category: "specialist",
    defaultBadge: "Specialist",
    accent: "#a855f7",
    hair: "#581c87",
    skin: "#e9b28d",
    outfit: "#6b21a8",
    accessory: "visor",
    hairStyle: "curls",
  },
];

export const SPECIALIST_POOL = CHARACTER_ROSTER.filter((c) => c.category === "specialist");

export function pickSpecialistCharacter(seed: string, index = 0): CharacterProfile {
  let hash = index;
  for (let i = 0; i < seed.length; i += 1) {
    hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
  }
  return SPECIALIST_POOL[hash % SPECIALIST_POOL.length];
}

export function pickVerifierCharacter(reviewerId: string): CharacterProfile {
  const low = reviewerId.toLowerCase();
  if (low.includes("number") || low.includes("arith")) return CHARACTER_ROSTER[2];
  if (low.includes("consist") || low.includes("logic")) return CHARACTER_ROSTER[3];
  return CHARACTER_ROSTER[1];
}

interface AvatarProps {
  profile: CharacterProfile;
  mood?: CharacterMood;
  size?: number;
}

export const CharacterAvatar: React.FC<AvatarProps> = ({
  profile,
  mood = "idle",
  size = 64,
}) => {
  const mouthPath =
    mood === "completed"
      ? "M27 36 Q32 41 37 36"
      : mood === "error"
      ? "M28 38 Q32 35 36 38"
      : mood === "working" || mood === "reviewing"
      ? "M29 36 Q32 38 35 36"
      : "M28 36 Q32 39 36 36";

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      className={`cf-avatar cf-mood-${mood}`}
      role="img"
      aria-label={profile.name}
    >
      {/* Subtle aura ring */}
      <circle cx="32" cy="32" r="30" fill={profile.accent} fillOpacity="0.14" stroke={profile.accent} strokeWidth="1.5" />

      {/* Back hair for long/bob/ponytail */}
      {(profile.hairStyle === "long" || profile.hairStyle === "bob") && (
        <path d="M17 24 C16 40, 20 46, 24 46 L40 46 C44 46, 48 40, 47 24 Z" fill={profile.hair} />
      )}
      {profile.hairStyle === "ponytail" && (
        <path d="M44 22 C52 24, 52 38, 46 42" stroke={profile.hair} strokeWidth="6" strokeLinecap="round" fill="none" />
      )}
      {profile.hairStyle === "bun" && (
        <circle cx="32" cy="11" r="6" fill={profile.hair} />
      )}

      {/* Shoulders / Outfit */}
      <path d="M16 58 C16 47, 23 44, 32 44 C41 44, 48 47, 48 58 Z" fill={profile.outfit} />
      <path d="M27 44 L32 49 L37 44" fill="#ffffff" fillOpacity="0.85" />

      {/* Scarf accessory */}
      {profile.accessory === "scarf" && (
        <path d="M22 44 Q32 50 42 44 L40 49 Q32 53 24 49 Z" fill={profile.accent} />
      )}

      {/* Head */}
      <circle cx="32" cy="29" r="13" fill={profile.skin} />

      {/* Front Hair */}
      {profile.hairStyle === "spiky" ? (
        <path d="M19 26 L22 15 L27 20 L32 13 L37 20 L42 15 L45 26 C41 21, 23 21, 19 26 Z" fill={profile.hair} />
      ) : profile.hairStyle === "curls" ? (
        <g fill={profile.hair}>
          <circle cx="22" cy="19" r="5" />
          <circle cx="29" cy="16" r="5.5" />
          <circle cx="36" cy="16" r="5.5" />
          <circle cx="42" cy="19" r="5" />
        </g>
      ) : (
        <path d="M19 27 C19 16, 45 16, 45 27 C40 22, 24 22, 19 27 Z" fill={profile.hair} />
      )}

      {/* Eyes */}
      <circle cx="27" cy="30" r="2.1" fill="#0f172a" />
      <circle cx="37" cy="30" r="2.1" fill="#0f172a" />
      <circle cx="27.7" cy="29.3" r="0.7" fill="#ffffff" />
      <circle cx="37.7" cy="29.3" r="0.7" fill="#ffffff" />

      {/* Blush */}
      <circle cx="23.5" cy="33" r="2" fill="#fb7185" fillOpacity="0.35" />
      <circle cx="40.5" cy="33" r="2" fill="#fb7185" fillOpacity="0.35" />

      {/* Mouth */}
      <path d={mouthPath} stroke="#7c2d12" strokeWidth="1.6" strokeLinecap="round" fill="none" />

      {/* Accessories */}
      {profile.accessory === "glasses" && (
        <g stroke="#e2e8f0" strokeWidth="1.3" fill="none">
          <rect x="23" y="27" width="8" height="6" rx="2" />
          <rect x="33" y="27" width="8" height="6" rx="2" />
          <line x1="31" y1="30" x2="33" y2="30" />
        </g>
      )}
      {profile.accessory === "visor" && (
        <rect x="22" y="26.5" width="20" height="6.5" rx="3" fill={profile.accent} fillOpacity="0.45" stroke={profile.accent} strokeWidth="1.2" />
      )}
      {profile.accessory === "goggles" && (
        <g>
          <rect x="21" y="17" width="9" height="6" rx="2.5" fill="#fbbf24" stroke="#78350f" strokeWidth="1.2" />
          <rect x="34" y="17" width="9" height="6" rx="2.5" fill="#fbbf24" stroke="#78350f" strokeWidth="1.2" />
          <line x1="30" y1="20" x2="34" y2="20" stroke="#78350f" strokeWidth="1.4" />
        </g>
      )}
      {profile.accessory === "headset" && (
        <g>
          <path d="M18 29 C18 16, 46 16, 46 29" stroke="#94a3b8" strokeWidth="2" fill="none" />
          <rect x="16" y="26" width="4" height="7" rx="2" fill={profile.accent} />
          <path d="M18 32 Q22 37 28 37" stroke="#94a3b8" strokeWidth="1.3" fill="none" />
        </g>
      )}
      {profile.accessory === "headphones" && (
        <g>
          <path d="M18 28 C18 15, 46 15, 46 28" stroke={profile.accent} strokeWidth="2.2" fill="none" />
          <rect x="16" y="25" width="4" height="8" rx="2" fill={profile.accent} />
          <rect x="44" y="25" width="4" height="8" rx="2" fill={profile.accent} />
        </g>
      )}
      {profile.accessory === "cap" && (
        <g>
          <path d="M19 22 C19 14, 45 14, 45 22 Z" fill={profile.accent} />
          <path d="M18 22 L48 22 C49 24, 43 25, 36 24 Z" fill={profile.outfit} />
        </g>
      )}
      {profile.accessory === "beanie" && (
        <path d="M19 22 C20 12, 44 12, 45 22 Z" fill={profile.accent} />
      )}
      {profile.accessory === "beret" && (
        <ellipse cx="34" cy="16" rx="13" ry="5" fill={profile.accent} transform="rotate(-10 34 16)" />
      )}

      {/* Status indicator dot */}
      {mood === "working" && <circle cx="52" cy="14" r="4" fill="#3b82f6" />}
      {mood === "reviewing" && <circle cx="52" cy="14" r="4" fill="#f59e0b" />}
      {mood === "completed" && <circle cx="52" cy="14" r="4" fill="#10b981" />}
      {mood === "error" && <circle cx="52" cy="14" r="4" fill="#ef4444" />}
    </svg>
  );
};
