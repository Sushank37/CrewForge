import React from "react";
import type { CharacterProfile } from "./CharacterSet";

export type SpritePose =
  | "idle"
  | "walking"
  | "working"
  | "reviewing"
  | "waiting"
  | "completed"
  | "error";

interface FullBodyCharacterProps {
  profile: CharacterProfile;
  pose: SpritePose;
  x: number;
  y: number;
  facing?: "left" | "right";
  roleLabel: string;
  bubble?: string | null;
  scale?: number;
}

export const FullBodyCharacter: React.FC<FullBodyCharacterProps> = ({
  profile,
  pose,
  x,
  y,
  facing = "right",
  roleLabel,
  bubble,
  scale = 1,
}) => {
  const flip = facing === "left" ? -1 : 1;
  const mouthPath =
    pose === "completed"
      ? "M-4 4 Q0 8 4 4"
      : pose === "error"
      ? "M-3.5 6 Q0 3 3.5 6"
      : pose === "working" || pose === "reviewing"
      ? "M-3 4.5 Q0 6 3 4.5"
      : "M-3.5 4.5 Q0 7 3.5 4.5";

  return (
    <g
      transform={`translate(${x}, ${y}) scale(${scale})`}
      className={`cf-sprite cf-pose-${pose}`}
      aria-label={`${profile.name} (${roleLabel}) - ${pose}`}
    >
      {/* Ground shadow */}
      <ellipse cx="0" cy="30" rx="16" ry="5" fill="#000000" fillOpacity="0.28" />

      {/* Animated body group */}
      <g className="cf-sprite-body" transform={`scale(${flip}, 1)`}>
        {/* Back hair for long / bob / ponytail / bun */}
        {(profile.hairStyle === "long" || profile.hairStyle === "bob") && (
          <path
            d="M-13 -16 C-15 2, -11 8, -7 8 L7 8 C11 8, 15 2, 13 -16 Z"
            fill={profile.hair}
          />
        )}
        {profile.hairStyle === "ponytail" && (
          <path
            d="M10 -18 C18 -16, 18 -3, 12 1"
            stroke={profile.hair}
            strokeWidth="5"
            strokeLinecap="round"
            fill="none"
          />
        )}
        {profile.hairStyle === "bun" && (
          <circle cx="0" cy="-29" r="5.5" fill={profile.hair} />
        )}

        {/* Legs & Shoes */}
        <g className="cf-leg-left">
          <rect x="-8" y="13" width="6" height="13" rx="2.5" fill="#1e293b" />
          <ellipse cx="-5" cy="27" rx="4.5" ry="2.6" fill="#0f172a" />
        </g>
        <g className="cf-leg-right">
          <rect x="2" y="13" width="6" height="13" rx="2.5" fill="#334155" />
          <ellipse cx="5" cy="27" rx="4.5" ry="2.6" fill="#0f172a" />
        </g>

        {/* Left Arm (back) */}
        <g className="cf-arm-left">
          <path
            d="M-10 -2 Q-17 4 -13 11"
            stroke={profile.outfit}
            strokeWidth="5.5"
            strokeLinecap="round"
            fill="none"
          />
          <circle cx="-13" cy="11.5" r="2.8" fill={profile.skin} />
        </g>

        {/* Torso / Jacket */}
        <path
          d="M-11 -4 C-11 -7, 11 -7, 11 -4 L12 14 C12 16, -12 16, -12 14 Z"
          fill={profile.outfit}
        />
        {/* Inner shirt collar */}
        <path d="M-4 -6 L0 1 L4 -6 Z" fill="#f8fafc" fillOpacity="0.92" />
        {/* Accent badge / trim */}
        <rect x="-8" y="2" width="4" height="3" rx="1" fill={profile.accent} />

        {/* Scarf accessory */}
        {profile.accessory === "scarf" && (
          <path
            d="M-9 -5 Q0 1 9 -5 L8 0 Q0 4 -8 0 Z"
            fill={profile.accent}
          />
        )}

        {/* Head */}
        <circle cx="0" cy="-14" r="11.5" fill={profile.skin} />

        {/* Front Hair */}
        {profile.hairStyle === "spiky" ? (
          <path
            d="M-11 -16 L-9 -27 L-4 -22 L0 -29 L4 -22 L9 -27 L11 -16 C7 -21, -7 -21, -11 -16 Z"
            fill={profile.hair}
          />
        ) : profile.hairStyle === "curls" ? (
          <g fill={profile.hair}>
            <circle cx="-8" cy="-22" r="4.5" />
            <circle cx="-2.5" cy="-25" r="4.8" />
            <circle cx="3.5" cy="-25" r="4.8" />
            <circle cx="8.5" cy="-22" r="4.5" />
          </g>
        ) : (
          <path
            d="M-11.5 -16 C-11.5 -27, 11.5 -27, 11.5 -16 C7 -21, -7 -21, -11.5 -16 Z"
            fill={profile.hair}
          />
        )}

        {/* Eyes */}
        <circle cx="-4.2" cy="-13" r="1.8" fill="#0f172a" />
        <circle cx="4.2" cy="-13" r="1.8" fill="#0f172a" />
        <circle cx="-3.6" cy="-13.6" r="0.6" fill="#ffffff" />
        <circle cx="4.8" cy="-13.6" r="0.6" fill="#ffffff" />

        {/* Blush & Mouth */}
        <circle cx="-7" cy="-10" r="1.6" fill="#fb7185" fillOpacity="0.32" />
        <circle cx="7" cy="-10" r="1.6" fill="#fb7185" fillOpacity="0.32" />
        <path
          d={mouthPath}
          transform="translate(0, -13)"
          stroke="#7c2d12"
          strokeWidth="1.4"
          strokeLinecap="round"
          fill="none"
        />

        {/* Accessories */}
        {profile.accessory === "glasses" && (
          <g stroke="#e2e8f0" strokeWidth="1.2" fill="none">
            <rect x="-8" y="-15.8" width="6.5" height="5" rx="1.6" />
            <rect x="1.5" y="-15.8" width="6.5" height="5" rx="1.6" />
            <line x1="-1.5" y1="-13.3" x2="1.5" y2="-13.3" />
          </g>
        )}
        {profile.accessory === "visor" && (
          <rect
            x="-9"
            y="-16"
            width="18"
            height="5.5"
            rx="2.6"
            fill={profile.accent}
            fillOpacity="0.48"
            stroke={profile.accent}
            strokeWidth="1.1"
          />
        )}
        {profile.accessory === "goggles" && (
          <g>
            <rect
              x="-9"
              y="-24.5"
              width="7.5"
              height="5"
              rx="2"
              fill="#fbbf24"
              stroke="#78350f"
              strokeWidth="1"
            />
            <rect
              x="1.5"
              y="-24.5"
              width="7.5"
              height="5"
              rx="2"
              fill="#fbbf24"
              stroke="#78350f"
              strokeWidth="1"
            />
            <line x1="-1.5" y1="-22" x2="1.5" y2="-22" stroke="#78350f" strokeWidth="1.2" />
          </g>
        )}
        {profile.accessory === "headset" && (
          <g>
            <path d="M-12 -14 C-12 -26, 12 -26, 12 -14" stroke="#94a3b8" strokeWidth="1.8" fill="none" />
            <rect x="-14" y="-17" width="3.5" height="6" rx="1.6" fill={profile.accent} />
            <path d="M-12 -11 Q-8 -7 -3 -7" stroke="#94a3b8" strokeWidth="1.2" fill="none" />
          </g>
        )}
        {profile.accessory === "headphones" && (
          <g>
            <path d="M-12 -15 C-12 -27, 12 -27, 12 -15" stroke={profile.accent} strokeWidth="2" fill="none" />
            <rect x="-14" y="-18" width="3.5" height="7" rx="1.6" fill={profile.accent} />
            <rect x="10.5" y="-18" width="3.5" height="7" rx="1.6" fill={profile.accent} />
          </g>
        )}
        {profile.accessory === "cap" && (
          <g>
            <path d="M-11 -20 C-11 -27, 11 -27, 11 -20 Z" fill={profile.accent} />
            <path d="M-11 -20 L14 -20 C15 -18.5, 10 -18, 4 -18.5 Z" fill={profile.outfit} />
          </g>
        )}
        {profile.accessory === "beanie" && (
          <path d="M-11 -20 C-10 -29, 10 -29, 11 -20 Z" fill={profile.accent} />
        )}
        {profile.accessory === "beret" && (
          <ellipse
            cx="2"
            cy="-25"
            rx="11.5"
            ry="4.2"
            fill={profile.accent}
            transform="rotate(-10 2 -25)"
          />
        )}

        {/* Right Arm (front) + Tool/Prop */}
        <g className="cf-arm-right">
          <path
            d="M10 -2 Q17 4 13 11"
            stroke={profile.outfit}
            strokeWidth="5.5"
            strokeLinecap="round"
            fill="none"
          />
          <circle cx="13" cy="11.5" r="2.8" fill={profile.skin} />
          {pose === "reviewing" && (
            <g transform="translate(15, 7)">
              <circle cx="4" cy="-3" r="4.5" fill="#bae6fd" fillOpacity="0.45" stroke="#f59e0b" strokeWidth="1.5" />
              <line x1="1" y1="0" x2="-2" y2="4" stroke="#f59e0b" strokeWidth="1.8" strokeLinecap="round" />
            </g>
          )}
        </g>
      </g>

      {/* Role nameplate */}
      <g transform="translate(0, 42)">
        <rect
          x="-54"
          y="-9"
          width="108"
          height="18"
          rx="9"
          fill="#0f172a"
          fillOpacity="0.86"
          stroke={profile.accent}
          strokeWidth="1.1"
        />
        <text
          x="0"
          y="3.5"
          textAnchor="middle"
          fontSize="9.5"
          fontWeight="600"
          fill="#f8fafc"
        >
          {roleLabel.length > 18 ? `${roleLabel.slice(0, 17)}…` : roleLabel}
        </text>
      </g>

      {/* Floating speech/action bubble */}
      {bubble && (
        <g transform="translate(0, -46)" className="cf-bubble-pop">
          <rect
            x="-66"
            y="-14"
            width="132"
            height="22"
            rx="11"
            fill="#0f172a"
            fillOpacity="0.94"
            stroke={profile.accent}
            strokeWidth="1.2"
          />
          <polygon points="-4,8 4,8 0,13" fill="#0f172a" />
          <text
            x="0"
            y="0.5"
            textAnchor="middle"
            fontSize="9.2"
            fontWeight="500"
            fill="#e2e8f0"
          >
            {bubble.length > 22 ? `${bubble.slice(0, 21)}…` : bubble}
          </text>
        </g>
      )}

      {/* Honest waiting badge */}
      {pose === "waiting" && (
        <g transform="translate(18, -28)">
          <circle cx="0" cy="0" r="8" fill="#f59e0b" />
          <path d="M0 -4 V0 L3 2" stroke="#0f172a" strokeWidth="1.6" strokeLinecap="round" fill="none" />
        </g>
      )}
    </g>
  );
};
