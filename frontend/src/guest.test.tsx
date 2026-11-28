import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { officeTrace, type GuestJob } from "./guest";
import { historicalAnswer } from "./conversation";

const anonymous = {
  authenticated: false, auth_required: true, google_enabled: false,
};

function json(data: unknown) {
  return new Response(JSON.stringify(data), {
    headers: { "Content-Type": "application/json" },
  });
}

const teamJob: GuestJob = {
  status: {
    task_id: "team", mode: "network", status: "completed", current_stage: "completed",
    provider: "server", model: "test-model", created_at: "2026-01-01T10:00:00Z",
    completed_at: "2026-01-01T10:00:02Z",
  },
  trace: {
    agents: [{ agent_id: "a1", role: "Research Analyst" }],
    assignments: [{ task_id: "a", title: "Calculate total", assigned_agent_id: "a1", status: "completed" }],
    messages: [{ message_id: "m", sender_id: "a1", recipient_id: "lead", content: "Total is ready." }],
    events: [{ seq: 1, event_type: "plan_created", message: "Real plan event" }],
    tool_executions: [{ tool_execution_id: "t", tool_name: "arithmetic", expression: "7*6", computed_result: "42", success: true, inputs_grounded: true }],
    review_decisions: [],
    honesty_gate: { passed: true, release_mode: "full_verified" },
  },
  result: {
    answer: "The verified team total is 42.",
    sources: [{ source_ref: "input", source_type: "user_prompt" }],
    evidence: [{ evidence_id: "e", source_ref: "input", source_type: "user_prompt", excerpt: "7 orders at 6 each" }],
    verification: {
      independent_verification_performed: true, honesty_gate_passed: true,
      release_mode: "full_verified", answer_complete: true,
    },
    unresolved_requirements: [],
    usage: { total_tokens: 123, model_calls: 7, actual_cost_usd: 0.002, usage_reported: true },
  },
};

describe("public guest experience", () => {
  let root: Root;
  let container: HTMLDivElement;

  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    sessionStorage.clear();
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
  });

  const findButton = (label: string) => Array.from(container.querySelectorAll("button"))
    .find((button) => button.textContent?.includes(label))!;

  async function click(label: string) {
    await act(async () => findButton(label).click());
  }

  it("opens the office without logging in and clearly reports paused execution", async () => {
    const fetchMock = vi.fn(async (path: string) => {
      if (path === "/api/auth/me") return json(anonymous);
      if (path === "/api/guest/chats") return json({ chats: [] });
      if (path === "/api/guest/session") return json({
        available: false, message: "Guest execution is paused while paid inference is disabled.",
      });
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    await act(async () => root.render(<App />));
    expect(container.textContent).toContain("Put an AI team");
    await click("Light mode");
    expect(document.documentElement.dataset.theme).toBe("light");
    await click("Continue as Guest");
    expect(container.querySelector("h1")?.textContent).toBe("What can we help you with?");
    expect(container.textContent).toContain("paid inference is disabled");
    expect(findButton("Send").disabled).toBe(true);
    expect(container.textContent).toContain("New Chat");
    expect(container.querySelector('a[href$="/mcp"]')).not.toBeNull();
    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual(["/api/auth/me", "/api/guest/session", "/api/guest/chats"]);

    await click("Private workspace");
    expect(container.querySelector('input[type="email"]')).not.toBeNull();
    await click("Back to CrewForge");
    expect(findButton("Continue as Guest")).toBeDefined();
  });

  it("follows the owned team run, requests the exact-input comparison and renders measured differences", async () => {
    sessionStorage.setItem("crewforge.guest.chat", "previous");
    const baseline = {
      ...teamJob,
      status: { ...teamJob.status, task_id: "single", mode: "baseline", completed_at: "2026-01-01T10:00:01Z" },
      trace: null,
      result: {
        ...teamJob.result,
        answer: "Single-agent answer.",
        verification: { independent_verification_performed: false, release_mode: "unverified_baseline" },
        usage: { ...teamJob.result!.usage, model_calls: 2, usage_reported: false, actual_cost_usd: null },
      },
    };
    const fetchMock = vi.fn(async (path: string, options?: RequestInit) => {
      if (path === "/api/auth/me") return json(anonymous);
      if (path === "/api/guest/session") return json({ available: true, message: null });
      if (path === "/api/guest/chats") return json({
        chats: [{ chat_id: "newer", title: "Another chat" }, { chat_id: "previous", title: "My task" }],
      });
      if (path === "/api/guest/chats/previous") return json({
        chat_id: "previous", title: "My task",
        turns: [{ message_id: "message", task_id: "team", prompt: "What is the total?",
          status: "completed", answer: teamJob.result!.answer, attachment_names: [], comparison_id: null }],
      });
      if (path === "/api/guest/jobs/team") return json(teamJob);
      if (path === "/api/guest/jobs/single") return json(baseline);
      if (path === "/api/guest/jobs") {
        expect(JSON.parse(options?.body as string)).toEqual({ compare_to: "team" });
        return json({ task_id: "single" });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    await act(async () => root.render(<App />));
    await click("Continue as Guest");
    expect(container.textContent).toContain("The verified team total is 42.");
    expect(container.textContent).not.toContain("7*6 = 42");
    expect(container.querySelector("table")).toBeNull();
    await click("Sources & details");
    expect(container.textContent).toContain("7*6 = 42");
    expect(container.textContent).toContain("7 orders at 6 each");
    expect(container.textContent).toContain("Total is ready.");
    await click("Compare with single agent");
    expect(container.textContent).toContain("Single-agent answer.");
    expect(container.textContent).toContain("Not performed");
    const cells = Array.from(container.querySelectorAll("tbody td")).map((cell) => cell.textContent);
    expect(cells).toEqual(["2.00 s", "1.00 s", "7", "2", "123", "—", "$0.0020000", "—"]);
  });

  it("retains scenario boundaries and uncertainty while hiding legacy trace formatting", () => {
    const view = historicalAnswer("**Conditional / Qualified Statements:**\n"
      + "- If approved, proceed. [clm_old@v1 (supported)]\n\n"
      + "**Calculations and Sources**\n- clm_old@v1 sources: doc_1.\n\n"
      + "**Unresolved / Withheld Items:**\n- The inspection date is unknown.");
    expect(view).toContain("If approved, proceed.");
    expect(view).toContain("The inspection date is unknown.");
    expect(view).not.toContain("clm_old");
  });

  it("maps the persisted events, assignments and messages without inferring a roster", () => {
    const trace = officeTrace(teamJob);
    expect(officeTrace(null)).toBeNull();
    expect(trace?.roster).toEqual(teamJob.trace?.agents);
    expect(trace?.tasks).toEqual(teamJob.trace?.assignments);
    expect(trace?.execution_events).toEqual(teamJob.trace?.events);
    expect(trace?.peer_messages).toEqual(teamJob.trace?.messages);
  });
});
