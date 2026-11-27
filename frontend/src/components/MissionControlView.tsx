import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { BenchmarkReport, TracePayload } from "../types";

interface MissionControlProps {
  initialRunId?: string | null;
}

type McTab = "live" | "replay" | "benchmark";
type DetailTab = "overview" | "topology" | "claims" | "timeline" | "gate";

export const MissionControlView: React.FC<MissionControlProps> = ({ initialRunId }) => {
  const [mcTab, setMcTab] = useState<McTab>(initialRunId ? "live" : "replay");
  const [detailTab, setDetailTab] = useState<DetailTab>("overview");
  const [trace, setTrace] = useState<TracePayload | null>(null);
  const [liveRuns, setLiveRuns] = useState<Array<{ run_id: string; status: string; model: string }>>([]);
  const [selectedLiveId, setSelectedLiveId] = useState<string>(initialRunId || "");
  const [replayFiles, setReplayFiles] = useState<Array<{ filename: string; run_id: string; status: string }>>([]);
  const [selectedReplay, setSelectedReplay] = useState<string>("");
  const [cursor, setCursor] = useState<number>(0);
  const [playing, setPlaying] = useState(false);
  const [selectedClaimKey, setSelectedClaimKey] = useState<string | null>(null);
  const [timelineFilter, setTimelineFilter] = useState<string>("all");
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialRunId) {
      setMcTab("live");
      setSelectedLiveId(initialRunId);
    }
  }, [initialRunId]);

  useEffect(() => {
    setError(null);
    setPlaying(false);
    setSelectedClaimKey(null);
    if (mcTab === "benchmark") {
      api.getBenchmark().then(setBenchmark).catch((e) => setError(e.message));
      return;
    }
    if (mcTab === "replay") {
      api
        .listReplayFiles()
        .then((res) => {
          setReplayFiles(res.files);
          const target = selectedReplay && res.files.some((f) => f.filename === selectedReplay)
            ? selectedReplay
            : res.files[0]?.filename || "";
          setSelectedReplay(target);
          if (target) return api.getReplayFile(target);
          return null;
        })
        .then((t) => {
          if (t) {
            setTrace(t);
            setCursor(t.execution_events.length);
          }
        })
        .catch((e) => setError(e.message));
      return;
    }
    api
      .listLiveRuns()
      .then((res) => {
        setLiveRuns(res.runs);
        const target =
          selectedLiveId && res.runs.some((r) => r.run_id === selectedLiveId)
            ? selectedLiveId
            : res.runs[0]?.run_id || "";
        setSelectedLiveId(target);
        if (target) return api.getLiveRun(target);
        setTrace(null);
        return null;
      })
      .then((t) => {
        if (t) {
          setTrace(t);
          setCursor(t.execution_events.length);
        }
      })
      .catch((e) => setError(e.message));
  }, [mcTab, selectedLiveId, selectedReplay]);

  useEffect(() => {
    if (!playing || !trace || mcTab !== "replay") return;
    const total = trace.execution_events.length;
    if (cursor >= total) {
      setPlaying(false);
      return;
    }
    const id = setInterval(() => {
      setCursor((c) => {
        if (c + 1 >= total) {
          setPlaying(false);
          return total;
        }
        return c + 1;
      });
    }, 500);
    return () => clearInterval(id);
  }, [playing, trace, mcTab, cursor]);

  const projected = useMemo(() => {
    if (!trace) return null;
    const allEvs = trace.execution_events || [];
    const full = mcTab !== "replay" || cursor >= allEvs.length;
    if (full) {
      return {
        events: allEvs,
        claims: trace.claims || [],
        verdicts: trace.review_verdicts || [],
        tools: trace.tool_executions || [],
        evidence: trace.evidence || [],
        gate: trace.honesty_gate || null,
        gateReached: Boolean(trace.honesty_gate || trace.final_result),
      };
    }
    const visibleEvents = allEvs.slice(0, cursor);
    const cutoff = visibleEvents.length ? visibleEvents[visibleEvents.length - 1].created_at || "" : "";
    const evTypes = new Set(visibleEvents.map((e) => e.event_type));
    const gateReached =
      evTypes.has("honesty_gate_evaluated") || evTypes.has("run_completed") || evTypes.has("run_failed");
    return {
      events: visibleEvents,
      claims: (trace.claims || []).filter((c) => cutoff && c.created_at && c.created_at <= cutoff),
      verdicts: (trace.review_verdicts || []).filter((v) => cutoff && v.created_at && v.created_at <= cutoff),
      tools: (trace.tool_executions || []).filter((t) => cutoff && t.created_at && t.created_at <= cutoff),
      evidence: (trace.evidence || []).filter((e) => cutoff && e.created_at && e.created_at <= cutoff),
      gate: gateReached ? trace.honesty_gate || null : null,
      gateReached,
    };
  }, [trace, mcTab, cursor]);

  const selectedClaim = useMemo(() => {
    if (!projected || !selectedClaimKey) return null;
    return projected.claims.find((c) => `${c.claim_id}@${c.version}` === selectedClaimKey) || null;
  }, [projected, selectedClaimKey]);

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1>Mission Control (Advanced Observability)</h1>
          <p className="muted">
            Inspect persisted traces, claim provenance, deterministic arithmetic, historical replay, and V2 benchmarks.
          </p>
        </div>
        <div className="input-row">
          {(["replay", "live", "benchmark"] as McTab[]).map((t) => (
            <button
              key={t}
              type="button"
              className={mcTab === t ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
              onClick={() => setMcTab(t)}
            >
              {t === "replay" ? "Historical Replay" : t === "live" ? "Live Session DB" : "Benchmark V2"}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {mcTab === "benchmark" && benchmark && (
        <div className="settings-grid">
          <section className="card">
            <h2>V2 Same-Model Comparison ({benchmark.model})</h2>
            <p className="muted">{benchmark.grading_method} — {benchmark.benchmark_note}</p>
            <div className="kpi-row">
              <div className="kpi-mini"><span>Baseline Mean</span><strong>{benchmark.summary.baseline_mean.toFixed(2)}/100</strong></div>
              <div className="kpi-mini"><span>CrewForge Mean</span><strong>{benchmark.summary.network_mean.toFixed(2)}/100</strong></div>
              <div className="kpi-mini"><span>Network Wins</span><strong>{benchmark.summary.network_wins} / 10</strong></div>
              <div className="kpi-mini"><span>Unsupported (B/N)</span><strong>{benchmark.summary.baseline_unsupported_assertions} vs {benchmark.summary.network_unsupported_assertions}</strong></div>
            </div>
            <table className="mc-table">
              <thead><tr><th>Domain</th><th>Baseline</th><th>Network</th></tr></thead>
              <tbody>
                {Object.entries(benchmark.domain_scores).map(([d, s]) => (
                  <tr key={d}>
                    <td><code>{d}</code></td>
                    <td>{s.baseline_mean.toFixed(2)}</td>
                    <td>{s.network_mean.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="card">
            <h2>David vs Goliath</h2>
            <span className="pill pill-verifier">{benchmark.david_vs_goliath.status}</span>
            <p className="muted">{benchmark.david_vs_goliath.detail}</p>
            <h3>Per-Case V2 Scores</h3>
            <table className="mc-table">
              <thead><tr><th>Case</th><th>Base</th><th>Net</th><th>Winner</th></tr></thead>
              <tbody>
                {benchmark.per_case.map((c) => (
                  <tr key={c.prompt_id}>
                    <td><code>{c.prompt_id}</code></td>
                    <td>{c.baseline.final_score}</td>
                    <td>{c.network.final_score}</td>
                    <td>{c.winner}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}

      {mcTab !== "benchmark" && (
        <>
          <div className="card mc-bar">
            {mcTab === "replay" ? (
              <div className="input-row" style={{ flexWrap: "wrap" }}>
                <select value={selectedReplay} onChange={(e) => setSelectedReplay(e.target.value)}>
                  {replayFiles.map((f) => (
                    <option key={f.filename} value={f.filename}>
                      {f.filename} ({f.run_id})
                    </option>
                  ))}
                </select>
                <button type="button" className="btn-ghost btn-sm" onClick={() => { setPlaying(false); setCursor(0); }}>
                  Restart
                </button>
                <button type="button" className="btn-primary btn-sm" onClick={() => setPlaying((p) => !p)}>
                  {playing ? "Pause" : "Play"}
                </button>
                <button
                  type="button"
                  className="btn-ghost btn-sm"
                  onClick={() => {
                    setPlaying(false);
                    setCursor((c) => Math.min(c + 1, trace?.execution_events.length || 0));
                  }}
                >
                  Step
                </button>
                <input
                  type="range"
                  min={0}
                  max={trace?.execution_events.length || 0}
                  value={cursor}
                  onChange={(e) => {
                    setPlaying(false);
                    setCursor(Number(e.target.value));
                  }}
                />
                <span className="muted">{cursor} / {trace?.execution_events.length || 0}</span>
              </div>
            ) : (
              <div className="input-row">
                <select value={selectedLiveId} onChange={(e) => setSelectedLiveId(e.target.value)}>
                  {liveRuns.length === 0 && <option value="">No live runs in judge DB</option>}
                  {liveRuns.map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.run_id} ({r.model} · {r.status})
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="input-row">
              {(["overview", "topology", "claims", "timeline", "gate"] as DetailTab[]).map((dt) => (
                <button
                  key={dt}
                  type="button"
                  className={detailTab === dt ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
                  onClick={() => setDetailTab(dt)}
                >
                  {dt}
                </button>
              ))}
            </div>
          </div>

          {!trace || !projected ? (
            <div className="card muted">No trace loaded yet.</div>
          ) : (
            <div className="card">
              {detailTab === "overview" && (
                <div className="kpi-row">
                  <div className="kpi-mini"><span>Run ID</span><strong>{trace.run_id}</strong></div>
                  <div className="kpi-mini"><span>Model</span><strong>{trace.provider} / {trace.model}</strong></div>
                  <div className="kpi-mini"><span>Status</span><strong>{trace.status}</strong></div>
                  <div className="kpi-mini"><span>Calls</span><strong>{trace.usage?.model_calls ?? "Unknown"}</strong></div>
                  <div className="kpi-mini"><span>Cost</span><strong>{trace.usage?.actual_cost_usd !== undefined ? `$${trace.usage.actual_cost_usd.toFixed(6)}` : "Unknown"}</strong></div>
                  <div className="kpi-mini"><span>Internal Req. Linkage (not grade)</span><strong>{projected.gate?.verification_summary?.requirement_coverage !== undefined ? `${(projected.gate.verification_summary.requirement_coverage * 100).toFixed(1)}%` : "Pending"}</strong></div>
                </div>
              )}

              {detailTab === "topology" && (
                <div>
                  <h3>Persisted Agent Roster ({trace.roster.length})</h3>
                  <div className="kpi-row">
                    {trace.roster.map((a) => (
                      <div key={a.agent_id} className="kpi-mini">
                        <span>{a.role}</span>
                        <strong>{a.agent_id}</strong>
                        <small className="muted">{a.specialty || a.model_override || trace.model}</small>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {detailTab === "claims" && (
                <div className="settings-grid">
                  <div>
                    <h3>Claims ({projected.claims.length})</h3>
                    {projected.claims.map((c) => {
                      const key = `${c.claim_id}@${c.version}`;
                      return (
                        <button
                          key={key}
                          type="button"
                          className={`claim-row-btn ${selectedClaimKey === key ? "selected" : ""}`}
                          onClick={() => setSelectedClaimKey(key)}
                        >
                          <code>{key}</code> <span className="pill">{c.status}</span>
                          <div>{c.statement}</div>
                        </button>
                      );
                    })}
                  </div>
                  <div>
                    <h3>Inspector</h3>
                    {!selectedClaim ? (
                      <div className="muted">Select a claim to inspect evidence, tools, and verdicts.</div>
                    ) : (
                      <div>
                        <p><strong>Dependencies:</strong> {[...(selectedClaim.depends_on_claim_ids || []), ...(selectedClaim.premise_refs || [])].join(", ") || "None"}</p>
                        {(selectedClaim.evidence_ids || []).map((eid) => {
                          const ev = projected.evidence.find((x) => x.evidence_id === eid);
                          return (
                            <div key={eid} className="sub-card">
                              <strong>{eid}</strong> [{ev?.source_type} · <code>{ev?.source_ref}</code>]
                              <div>{ev?.excerpt}</div>
                            </div>
                          );
                        })}
                        {(selectedClaim.tool_execution_ids || []).map((tid) => {
                          const tl = projected.tools.find((x) => x.tool_execution_id === tid);
                          return (
                            <div key={tid} className="sub-card">
                              <code>{tl?.expression}</code> = <strong>{tl?.computed_result ?? "none"}</strong> {tl?.units || ""} (grounded={String(tl?.inputs_grounded)})
                            </div>
                          );
                        })}
                        {projected.verdicts
                          .filter((v) => v.claim_id === selectedClaim.claim_id)
                          .map((v) => (
                            <div key={v.verdict_id} className="sub-card">
                              <strong>{v.reviewer_id}</strong> (v{v.claim_version}): {v.decision || v.verdict} — {v.rationale}
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {detailTab === "timeline" && (
                <div>
                  <div className="input-row" style={{ marginBottom: 8 }}>
                    {["all", "planning", "task_execution", "tool_calls", "review", "revision", "honesty_gate"].map((cat) => (
                      <button
                        key={cat}
                        type="button"
                        className={timelineFilter === cat ? "btn-primary btn-sm" : "btn-ghost btn-sm"}
                        onClick={() => setTimelineFilter(cat)}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                  <div className="mc-timeline">
                    {projected.events
                      .filter((e) => timelineFilter === "all" || e.category === timelineFilter)
                      .map((ev) => (
                        <div key={ev.seq} className="mc-ev">
                          <code>#{ev.seq} {(ev.created_at || "").slice(11, 19)}</code>
                          <strong>{ev.event_type}</strong>
                          <span>{ev.message}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {detailTab === "gate" && (
                <div>
                  {!projected.gateReached ? (
                    <div className="muted">Pending replay of honesty_gate_evaluated event.</div>
                  ) : (
                    <div>
                      <p>
                        <strong>Release mode:</strong> <code>{projected.gate?.release_mode}</code> ·{" "}
                        <strong>Passed:</strong> <code>{String(projected.gate?.passed)}</code>
                      </p>
                      <pre className="code-pre">{trace.final_result?.answer || "(No answer released)"}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};
