import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { CharacterGallery } from "./components/CharacterGallery";
import { ChatView } from "./components/ChatView";
import { McpView } from "./components/McpView";
import { MissionControlView } from "./components/MissionControlView";
import { SettingsView } from "./components/SettingsView";
import type { AppView, ChatSummary, ThemeMode } from "./types";

export const App: React.FC = () => {
  const [view, setView] = useState<AppView>("chat");
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [reducedMotion, setReducedMotion] = useState<boolean>(false);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [inspectRunId, setInspectRunId] = useState<string | null>(null);

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
          prev && res.chats.some((c) => c.chat_id === prev) ? prev : res.chats[0].chat_id
        );
      }
    } catch {
      /* backend starting */
    }
  }, []);

  useEffect(() => {
    loadChats();
  }, [loadChats]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    document.documentElement.classList.toggle("reduced-motion", reducedMotion);
  }, [reducedMotion]);

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

      <main className="main-content">
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
