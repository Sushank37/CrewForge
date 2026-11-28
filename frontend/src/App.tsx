import React, { useCallback, useEffect, useState } from "react";
import { api, setSessionToken } from "./api";
import { AuthPage } from "./components/AuthPage";
import { CharacterGallery } from "./components/CharacterGallery";
import { ChatView } from "./components/ChatView";
import { McpView } from "./components/McpView";
import { MissionControlView } from "./components/MissionControlView";
import { GuestExperience, PublicLanding } from "./components/PublicExperience";
import { SettingsView } from "./components/SettingsView";
import type { AppView, AuthMeResponse, ChatSummary, ThemeMode } from "./types";

export const App: React.FC = () => {
  const [view, setView] = useState<AppView>("chat");
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [reducedMotion, setReducedMotion] = useState<boolean>(false);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [inspectRunId, setInspectRunId] = useState<string | null>(null);
  const [authInfo, setAuthInfo] = useState<AuthMeResponse | null>(null);
  const [showSignIn, setShowSignIn] = useState<boolean>(false);
  const [guestOpen, setGuestOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState<boolean>(false);

  const resetAccountState = useCallback(() => {
    setChats([]);
    setActiveChatId(null);
    setInspectRunId(null);
    setView("chat");
    setProfileOpen(false);
    setSessionToken(null);
  }, []);

  const loadChats = useCallback(async () => {
    try {
      const res = await api.listChats();
      setChats(res.chats);
      if (res.chats.length === 0) {
        const created = await api.createChat("New Chat");
        setChats([created]);
        setActiveChatId(created.chat_id);
      } else {
        setActiveChatId((prev) =>
          prev && res.chats.some((c) => c.chat_id === prev) ? prev
            : res.chats.find((c) => c.chat_id === sessionStorage.getItem("crewforge.private.chat"))?.chat_id
              ?? res.chats[0].chat_id
        );
      }
    } catch {
      /* unauthenticated or backend starting */
    }
  }, []);

  const refreshAuthAndChats = useCallback(async () => {
    try {
      const me = await api.authMe();
      setAuthInfo((prev) => {
        if (prev?.user_id && prev.user_id !== me.user_id) {
          resetAccountState();
        }
        return me;
      });
      if (me.authenticated || !me.auth_required) {
        setShowSignIn(false);
        await loadChats();
      }
    } catch {
      /* backend starting */
    }
  }, [loadChats, resetAccountState]);

  useEffect(() => {
    void refreshAuthAndChats();
  }, [refreshAuthAndChats]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    document.documentElement.classList.toggle("reduced-motion", reducedMotion);
  }, [reducedMotion]);

  useEffect(() => {
    if (activeChatId) sessionStorage.setItem("crewforge.private.chat", activeChatId);
  }, [activeChatId]);

  const handleNewChat = async () => {
    const created = await api.createChat("New Chat");
    setChats((prev) => [created, ...prev]);
    setActiveChatId(created.chat_id);
    setView("chat");
  };

  const handleDeleteChat = async (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    await api.deleteChat(chatId);
    await loadChats();
  };

  const handleInspectRun = (runId: string) => {
    setInspectRunId(runId);
    setView("mission-control");
  };

  const handleAuthenticated = async () => {
    resetAccountState();
    await refreshAuthAndChats();
  };

  const handleLogout = async () => {
    try {
      await api.logout();
    } catch {
      /* ignore */
    }
    resetAccountState();
    await refreshAuthAndChats();
  };

  const publicProps = {
    theme,
    onToggleTheme: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    onGuest: () => setGuestOpen(true),
    onHome: () => { setGuestOpen(false); setShowSignIn(false); },
    onPrivate: () => { setGuestOpen(false); setShowSignIn(true); },
  };

  if (guestOpen) return <GuestExperience {...publicProps} />;

  if (showSignIn) {
    return (
      <AuthPage
        authInfo={authInfo ?? { authenticated: false, auth_required: true, google_enabled: false }}
        theme={theme}
        onToggleTheme={publicProps.onToggleTheme}
        onAuthenticated={handleAuthenticated}
        onCancel={publicProps.onHome}
        cancelLabel={authInfo?.auth_required === false ? "Back to local workspace" : "Back to CrewForge"}
      />
    );
  }

  if (!authInfo || (authInfo.auth_required && !authInfo.authenticated)) {
    return <PublicLanding {...publicProps} />;
  }

  const accountKey = authInfo?.user_id || "local";
  const activeUser = authInfo?.user;

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary Navigation">
        <div className="sidebar-brand">
          <div className="brand-badge">CF</div>
          <div>
            <div className="brand-title">CrewForge</div>
            <div className="brand-sub">AI Team Workspace</div>
          </div>
        </div>

        <button type="button" className="btn-new-chat" onClick={handleNewChat}>
          + New Chat
        </button>

        <nav className="side-nav">
          <button
            type="button"
            className={`nav-item ${view === "chat" ? "active" : ""}`}
            onClick={() => setView("chat")}
          >
            💬 Chats
          </button>
          <button
            type="button"
            className={`nav-item ${view === "mission-control" ? "active" : ""}`}
            onClick={() => setView("mission-control")}
          >
            📡 Mission Control
          </button>
          <button
            type="button"
            className={`nav-item ${view === "characters" ? "active" : ""}`}
            onClick={() => setView("characters")}
          >
            🎭 Character Set (16)
          </button>
          <button
            type="button"
            className={`nav-item ${view === "mcp" ? "active" : ""}`}
            onClick={() => setView("mcp")}
          >
            🔌 MCP Connection
          </button>
          <button
            type="button"
            className={`nav-item ${view === "settings" ? "active" : ""}`}
            onClick={() => setView("settings")}
          >
            ⚙️ Settings
          </button>
        </nav>

        <div className="recent-section">
          <div className="recent-label">Recent Chats</div>
          <div className="recent-list">
            {chats.map((c) => (
              <div
                key={c.chat_id}
                role="button"
                tabIndex={0}
                className={`recent-item ${view === "chat" && activeChatId === c.chat_id ? "active" : ""}`}
                onClick={() => {
                  setActiveChatId(c.chat_id);
                  setView("chat");
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    setActiveChatId(c.chat_id);
                    setView("chat");
                  }
                }}
              >
                <span className="recent-title">{c.title}</span>
                <button
                  type="button"
                  className="recent-del"
                  title="Delete chat"
                  onClick={(e) => handleDeleteChat(e, c.chat_id)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          {authInfo?.authenticated && activeUser ? (
            <div className="profile-box">
              <button
                type="button"
                className="profile-trigger"
                onClick={() => setProfileOpen((o) => !o)}
              >
                <span className="profile-avatar">
                  {(activeUser.display_name || activeUser.email || "U")[0].toUpperCase()}
                </span>
                <span className="profile-meta">
                  <span className="profile-name">
                    {activeUser.display_name || activeUser.email || activeUser.user_id}
                  </span>
                  <span className="profile-sub">{activeUser.email || activeUser.user_id}</span>
                </span>
              </button>
              {profileOpen && (
                <div className="profile-menu">
                  <div className="profile-detail">
                    <strong>Account ID</strong>
                    <code>{activeUser.user_id}</code>
                  </div>
                  <div className="profile-detail">
                    <strong>Status</strong>
                    <span>
                      {activeUser.email_verified ? "Verified" : "Provisioned"}
                      {activeUser.google_linked ? " · Google" : ""}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="btn-ghost btn-sm"
                    style={{ width: "100%", marginTop: 6 }}
                    onClick={() => void handleLogout()}
                  >
                    Log out
                  </button>
                </div>
              )}
            </div>
          ) : (
            authInfo && (
              <button
                type="button"
                className="btn-ghost btn-sm"
                style={{ width: "100%", marginBottom: 6 }}
                onClick={() => setShowSignIn(true)}
              >
                Sign in with Google / Email
              </button>
            )
          )}
          <button
            type="button"
            className="btn-ghost btn-sm"
            style={{ width: "100%" }}
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode"}
          </button>
        </div>
      </aside>

      <main className="main-content" key={accountKey}>
        {view === "chat" && (
          <ChatView
            chatId={activeChatId}
            onChatUpdated={loadChats}
            onInspectRun={handleInspectRun}
          />
        )}
        {view === "mission-control" && <MissionControlView initialRunId={inspectRunId} />}
        {view === "characters" && <CharacterGallery />}
        {view === "mcp" && <McpView />}
        {view === "settings" && (
          <SettingsView
            theme={theme}
            onThemeChange={setTheme}
            reducedMotion={reducedMotion}
            onReducedMotionChange={setReducedMotion}
          />
        )}
      </main>
    </div>
  );
};

export default App;
