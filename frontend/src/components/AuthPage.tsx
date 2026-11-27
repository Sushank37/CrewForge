import React, { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { AuthMeResponse, ThemeMode } from "../types";

declare global {
  interface Window {
    google?: {
      accounts?: {
        id?: {
          initialize: (opts: {
            client_id: string;
            callback: (resp: { credential?: string }) => void;
          }) => void;
          renderButton: (
            el: HTMLElement,
            opts: Record<string, unknown>
          ) => void;
        };
      };
    };
  }
}

interface AuthPageProps {
  authInfo: AuthMeResponse;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onAuthenticated: () => Promise<void>;
  onCancelLocal?: () => void;
}

export const AuthPage: React.FC<AuthPageProps> = ({
  authInfo,
  theme,
  onToggleTheme,
  onAuthenticated,
  onCancelLocal,
}) => {
  const [step, setStep] = useState<"email" | "code">("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const googleBtnRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const clientId = authInfo.google_client_id;
    if (!clientId || !googleBtnRef.current) return;

    let cancelled = false;
    const initGsi = () => {
      if (cancelled || !window.google?.accounts?.id || !googleBtnRef.current) return;
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: async (resp) => {
          if (!resp?.credential) {
            setErr("Google sign-in did not return a credential.");
            return;
          }
          setBusy(true);
          setErr(null);
          try {
            await api.loginWithGoogle(resp.credential);
            await onAuthenticated();
          } catch (e) {
            setErr(e instanceof Error ? e.message : String(e));
          } finally {
            setBusy(false);
          }
        },
      });
      googleBtnRef.current.innerHTML = "";
      window.google.accounts.id.renderButton(googleBtnRef.current, {
        theme: theme === "dark" ? "filled_black" : "outline",
        size: "large",
        width: 340,
        text: "continue_with",
        shape: "pill",
      });
    };

    if (window.google?.accounts?.id) {
      initGsi();
      return () => {
        cancelled = true;
      };
    }

    const existing = document.querySelector<HTMLScriptElement>(
      'script[src="https://accounts.google.com/gsi/client"]'
    );
    if (existing) {
      existing.addEventListener("load", initGsi);
      return () => {
        cancelled = true;
        existing.removeEventListener("load", initGsi);
      };
    }

    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = initGsi;
    document.head.appendChild(script);
    return () => {
      cancelled = true;
    };
  }, [authInfo.google_client_id, theme, onAuthenticated]);

  const handleRequestCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || busy) return;
    setBusy(true);
    setErr(null);
    setInfoMsg(null);
    try {
      const res = await api.requestEmailCode(email.trim());
      setEmail(res.email);
      setStep("code");
      setInfoMsg(
        `A 6-digit verification code was sent to ${res.email}. It expires in 10 minutes.`
      );
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim() || busy) return;
    setBusy(true);
    setErr(null);
    try {
      await api.verifyEmailCode(email.trim(), code.trim());
      await onAuthenticated();
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <div className="brand-badge" style={{ width: 42, height: 42, fontSize: 16 }}>
            CF
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: 20 }}>Sign in to CrewForge</h1>
            <p className="muted" style={{ margin: "2px 0 0", fontSize: 13 }}>
              Private workspace, persistent chat history, and personal MCP tokens
            </p>
          </div>
        </div>

        {err && <div className="alert-error">{err}</div>}
        {infoMsg && <div className="alert-info">{infoMsg}</div>}

        <div className="auth-section">
          {authInfo.google_client_id ? (
            <div ref={googleBtnRef} className="google-btn-slot" />
          ) : (
            <button
              type="button"
              className="btn-ghost auth-provider-btn"
              disabled
              title="Set CREWFORGE_GOOGLE_CLIENT_ID to enable Google Sign-In"
            >
              Continue with Google (Configure CREWFORGE_GOOGLE_CLIENT_ID)
            </button>
          )}
        </div>

        <div className="auth-divider">
          <span>or</span>
        </div>

        {step === "email" ? (
          <form onSubmit={handleRequestCode} className="auth-form">
            <label className="field">
              <span>Email address</span>
              <input
                type="email"
                required
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={busy}
              />
            </label>
            <button type="submit" className="btn-primary auth-submit-btn" disabled={busy}>
              {busy ? "Sending code..." : "Continue with Email"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleVerifyCode} className="auth-form">
            <label className="field">
              <span>Verification code for {email}</span>
              <input
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                placeholder="6-digit code"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                disabled={busy}
              />
            </label>
            <button type="submit" className="btn-primary auth-submit-btn" disabled={busy}>
              {busy ? "Verifying..." : "Verify & Continue"}
            </button>
            <div className="auth-links">
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={() => {
                  setStep("email");
                  setCode("");
                  setErr(null);
                  setInfoMsg(null);
                }}
                disabled={busy}
              >
                Use different email
              </button>
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={(e) => void handleRequestCode(e)}
                disabled={busy}
              >
                Resend code
              </button>
            </div>
          </form>
        )}

        <div className="auth-footer">
          <button type="button" className="btn-ghost btn-xs" onClick={onToggleTheme}>
            {theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode"}
          </button>
          {onCancelLocal && (
            <button type="button" className="btn-ghost btn-xs" onClick={onCancelLocal}>
              Back to local workspace
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
