// playground/web/src/pages/AgentView.tsx
import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchConfig,
  fetchPlaybooks,
  type PlaybookInfo,
  type UnpodConfig,
} from "../config";
import { SupervoiceClient } from "../client/SupervoiceClient";
import { SupervoiceWSTransport } from "../transport/SupervoiceWSTransport";
import { Composer } from "../components/Composer";
import { DashboardPanel } from "../components/DashboardPanel";
import { ConversationPanel } from "../components/ConversationPanel";
import { PlaybookList } from "../components/PlaybookList";
import { MetricsPanel } from "../components/MetricsPanel";
import { PipelinePanel } from "../components/PipelinePanel";
import { VoiceProfilePanel } from "../components/VoiceProfilePanel";
import { EventsLog } from "../components/EventsLog";
import { EditPanel } from "../components/EditPanel";
import { ChatPanel } from "../components/ChatPanel";
import { StatsPanel } from "../components/StatsPanel";
import { TopBar } from "../components/TopBar";
import {
  clockTime,
  EMPTY_METRICS,
  type AppState,
  type LLMCallEvent,
  type LogEntry,
  type MetricSnapshot,
  type SessionInfo,
  type TurnCompleteEvent,
  type Turn,
} from "../types";
import { isTurnActive, type ConvState } from "../state/convState";
import { usePlaybookEditor } from "../hooks/usePlaybookEditor";

// Left work pane: preview the running plan, edit it, inspect the call.
type Tab =
  | "preview"
  | "edit"
  | "conversation"
  | "metrics"
  | "traces"
  | "events";
// Right control pane: build with the AI agent, watch stats, pick a playbook.
type RightTab = "chat" | "stats" | "playbooks";

const LEFT_TABS: Array<{ id: Tab; label: string }> = [
  { id: "preview", label: "Preview" },
  { id: "edit", label: "Edit" },
  { id: "conversation", label: "Conversation" },
  { id: "metrics", label: "Metrics" },
  { id: "traces", label: "Traces" },
  { id: "events", label: "Events" },
];
const RIGHT_TABS: Array<{ id: RightTab; label: string }> = [
  { id: "chat", label: "Chat" },
  { id: "stats", label: "Stats" },
  { id: "playbooks", label: "Playbooks" },
];

function startResize(
  e: React.MouseEvent,
  axis: "x" | "y",
  current: number,
  set: (v: number) => void,
  direction: 1 | -1,
  min: number,
  max: number,
) {
  e.preventDefault();
  const start = axis === "x" ? e.clientX : e.clientY;
  const onMove = (ev: MouseEvent) => {
    const pos = axis === "x" ? ev.clientX : ev.clientY;
    set(Math.max(min, Math.min(max, current + (pos - start) * direction)));
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);

/** One-line, human-readable summary of a side-channel event for the log. */
function describe(type: string, d: Record<string, unknown>): string {
  switch (type) {
    case "connected":
      return "audio websocket open";
    case "disconnected":
      return String(d.reason ?? "connection closed");
    case "session":
      return `${d.transport ?? "ws"} · ${d.session_id ?? "no id"}`;
    case "ready":
      return `agent ${d.agent ?? ""} ready`;
    case "user_turn":
      return `"${d.text ?? ""}"`;
    case "agent_turn":
      return `"${d.text ?? ""}"`;
    case "llm_call":
      return `T${d.turn_id ?? "?"} ${d.node_id ?? "?"} ${d.call_type ?? "call"} · ${
        d.latency_ms ?? "—"
      }ms`;
    case "turn_complete":
      return `T${d.turn_id ?? "?"} ${d.from_node ?? "?"} → ${
        d.to_node ?? "?"
      } · ttfa=${d.ttfa_ms ?? "—"}ms · ${String(d.agent_text ?? "").slice(0, 32)}`;
    case "metric":
      return `ttfa=${d.ttfa_ms ?? "—"}ms turns=${d.turns ?? 0} cost=$${d.cost_usd_so_far ?? 0}`;
    case "interruption":
      return "user interrupted the agent";
    case "state":
      return `${d.state ?? "?"}${d.turn_id != null ? ` · T${d.turn_id}` : ""}`;
    case "flow_node_changed":
      return `node → ${d.node_id ?? "?"}`;
    case "error":
      return `[${d.severity ?? "error"}] ${d.code ?? ""}: ${d.message ?? ""}`;
    default:
      return JSON.stringify(d);
  }
}

export function AgentView() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [config, setConfig] = useState<UnpodConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [selectedVoiceProfile, setSelectedVoiceProfile] = useState("");
  const [tab, setTab] = useState<Tab>("preview");
  const [rightPaneTab, setRightPaneTab] = useState<RightTab>("playbooks");
  // The single source of truth for conversation state — written ONLY by the WS
  // transport's onState (plus the disconnect/connect reset). No timers.
  const [convState, setConvState] = useState<ConvState>("idle");
  // Latest turn text, surfaced in the pipeline readout (gated on convState).
  const [lastUserText, setLastUserText] = useState("");
  const [lastAgentText, setLastAgentText] = useState("");

  // Pipeline node readout (Preview tab) — fed by the worker's state event.
  const [currentNode, setCurrentNode] = useState<string | null>(null);

  const [playbooks, setPlaybooks] = useState<PlaybookInfo[]>([]);
  const [activePlaybook, setActivePlaybook] = useState("");
  // The playbook editor (YAML buffer, validation, draft/save/publish) — shared
  // by the Edit tab and the TopBar controls so they never diverge.
  const editor = usePlaybookEditor(activePlaybook);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [metrics, setMetrics] = useState<MetricSnapshot>(EMPTY_METRICS);
  const [llmCalls, setLlmCalls] = useState<LLMCallEvent[]>([]);
  const [turnTimings, setTurnTimings] = useState<TurnCompleteEvent[]>([]);
  const [events, setEvents] = useState<LogEntry[]>([]);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [agentReady, setAgentReady] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [botLevel, setBotLevel] = useState(0);

  const [rightColWidth, setRightColWidth] = useState(360);

  const clientRef = useRef<SupervoiceClient | null>(null);
  const logId = useRef(0);
  const activePlaybookRef = useRef(""); // current playbook selection, read at connect
  const connectingRef = useRef(false); // synchronous re-entrancy guard
  const mounted = useRef(true);

  useEffect(() => {
    Promise.all([fetchConfig(), fetchPlaybooks()])
      .then(([cfg, pb]) => {
        setConfig(cfg);
        setSelectedVoiceProfile(cfg.voice_profiles[0]?.id ?? "");
        setPlaybooks(pb.playbooks);
        setActivePlaybook(pb.active ?? pb.playbooks[0]?.id ?? "");
      })
      .catch((err) => setConfigError((err as Error).message));
  }, []);

  // Keep the ref in sync so the connect() closure reads the live selection.
  useEffect(() => {
    activePlaybookRef.current = activePlaybook;
  }, [activePlaybook]);

  // Tear down the live client (mic + WebSockets) if the view unmounts.
  useEffect(
    () => () => {
      mounted.current = false;
      void clientRef.current?.disconnect();
      clientRef.current = null;
    },
    [],
  );

  // Level meters decay toward 0 between audio frames.
  useEffect(() => {
    if (appState !== "active") return;
    const id = window.setInterval(() => {
      setMicLevel((l) => (l > 0.01 ? l * 0.65 : 0));
      setBotLevel((l) => (l > 0.01 ? l * 0.6 : 0));
    }, 100);
    return () => window.clearInterval(id);
  }, [appState]);

  const pushEvent = useCallback((type: string, data: Record<string, unknown>) => {
    setEvents((prev) => {
      const entry: LogEntry = {
        id: logId.current++,
        time: clockTime(),
        type,
        detail: describe(type, data),
      };
      const next = [...prev, entry];
      return next.length > 500 ? next.slice(-500) : next;
    });
  }, []);

  const addTurn = useCallback((role: Turn["role"], text: string) => {
    if (!mounted.current) return;
    setTurns((prev) => [...prev, { role, text, time: clockTime() }]);
  }, []);

  const playbookLabel = useCallback(
    (id: string) => playbooks.find((p) => p.id === id)?.label ?? id,
    [playbooks],
  );

  const connect = useCallback(async () => {
    if (connectingRef.current || appState === "active") return; // re-entrancy guard
    connectingRef.current = true;

    const stale = clientRef.current;
    clientRef.current = null;
    if (stale) await stale.disconnect();

    setAppState("connecting");
    setTurns([]);
    setMetrics(EMPTY_METRICS);
    setLlmCalls([]);
    setTurnTimings([]);
    setEvents([]);
    setSession(null);
    setAgentReady(false);
    setCurrentNode(null);
    setMicLevel(0);
    setBotLevel(0);
    setConvState("idle");
    setLastUserText("");
    setLastAgentText("");

    // Playbook is the only engine the UI runs (Flows dropped from the UI). The
    // mode + file ride the session request → worker builds the playbook engine.
    const transport = new SupervoiceWSTransport(
      undefined,
      selectedVoiceProfile,
      undefined,
      "playbook",
      activePlaybookRef.current,
    );
    const client = new SupervoiceClient({ transport });
    clientRef.current = client;

    client.onAny(pushEvent);
    client.onMicLevel((l) => {
      if (!mounted.current) return;
      setMicLevel((cur) => Math.max(cur, l));
    });
    client.onBotLevel((l) => {
      // botLevel is waveform amplitude ONLY — never conversation state.
      if (!mounted.current) return;
      setBotLevel((cur) => Math.max(cur, l));
    });
    // The single writer of convState: worker-authored state arrives down the WS
    // transport (Sink A). No timers, no botLevel threshold — see the design's
    // single-source invariants. The disconnect/connect resets below are the
    // only other writers.
    client.onState((s) => {
      if (!mounted.current) return;
      setConvState(s);
      // New-turn boundary: drop the prior turn's readout text so the pipeline
      // shows a generic label until the matching user_turn/agent_turn text
      // lands. convState (media) precedes the turn text (agent layer), so
      // without this the readout would show the PREVIOUS turn's text as if it
      // were current (design's "state vs. turn-text ordering" edge case).
      if (s === "listening" || s === "interrupted") setLastUserText("");
      if (s === "listening" || s === "interrupted" || s === "thinking") {
        setLastAgentText("");
      }
    });

    client.on("connected", () => {
      setAppState("active");
    });
    client.on("disconnected", () => {
      setAppState("disconnected");
      setAgentReady(false);
      setConvState("idle");
    });
    client.on("ready", () => {
      setAgentReady(true);
      // The selected playbook is applied on the worker BEFORE the opening turn
      // (sent as call metadata via /playground/sessions → ctx.data), so the
      // greeting already matches the selection — no post-start switch here.
    });
    // Barge-in is reflected by convState (interrupted → listening) from the
    // worker; the side-channel "interruption" still surfaces in the EVENTS log
    // via onAny. No timer-driven UI reset here.
    client.on("session", (d) => setSession(d as SessionInfo));
    client.on("flow_node_changed", (d) =>
      setCurrentNode(d.node_id != null ? String(d.node_id) : null),
    );
    client.on("user_turn", (d) => {
      const text = String(d.text ?? "");
      addTurn("user", text);
      // Turn text only — the pipeline lights from convState, not from this.
      setLastUserText(text);
    });
    client.on("agent_turn", (d) => {
      const text = String(d.text ?? "");
      addTurn("agent", text);
      setLastAgentText(text);
    });
    client.on("metric", (d) =>
      setMetrics({
        ttfa_ms: num(d.ttfa_ms),
        asr_p95_ms: num(d.asr_p95_ms),
        tts_p95_ms: num(d.tts_p95_ms),
        turns: num(d.turns) ?? 0,
        cost_usd_so_far: num(d.cost_usd_so_far),
      }),
    );
    client.on("llm_call", (d) => {
      setLlmCalls((prev) => {
        const next = [...prev, d as unknown as LLMCallEvent];
        return next.length > 500 ? next.slice(-500) : next;
      });
    });
    client.on("turn_complete", (d) => {
      setTurnTimings((prev) => {
        const incoming = d as unknown as TurnCompleteEvent;
        const idx = prev.findIndex((item) => item.turn_id === incoming.turn_id);
        const next =
          idx === -1
            ? [...prev, incoming]
            : prev.map((item) =>
                item.turn_id === incoming.turn_id
                  ? {
                      ...item,
                      ...incoming,
                      // Preserve non-null node data — TurnMetricsEvent has null nodes
                      from_node: incoming.from_node ?? item.from_node,
                      to_node: incoming.to_node ?? item.to_node,
                      // Keep higher llm count and total from either event
                      llm_call_count:
                        incoming.llm_call_count || item.llm_call_count,
                      llm_total_ms: incoming.llm_total_ms ?? item.llm_total_ms,
                    }
                  : item,
              );
        return next.length > 200 ? next.slice(-200) : next;
      });
    });
    client.on("error", (d) => {
      addTurn("system", `Error: ${String(d.message ?? "unknown")}`);
      setAppState("idle");
      setConvState("idle");
    });

    try {
      await client.connect();
    } catch (err) {
      addTurn("system", `Connect failed: ${(err as Error).message}`);
      pushEvent("error", { message: (err as Error).message });
      setAppState("idle");
      clientRef.current = null;
    } finally {
      connectingRef.current = false;
    }
  }, [appState, selectedVoiceProfile, addTurn, pushEvent]);

  const disconnect = useCallback(async () => {
    await clientRef.current?.disconnect();
    clientRef.current = null;
  }, []);

  const selectPlaybook = useCallback((playbookId: string) => {
    // Playbooks are a single compound runtime — they can't switch live, so the
    // selection just applies on the next Connect (the list locks while live).
    setActivePlaybook(playbookId);
  }, []);

  const handleExport = useCallback(() => {
    // Download the current editor YAML as a file (browser-only, no backend).
    const blob = new Blob([editor.yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activePlaybook || "playbook"}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  }, [editor.yaml, activePlaybook]);

  if (configError) {
    return (
      <div className="loading loading--error">
        Could not reach the harness: {configError}
        <br />
        Is it running on this port? (<code>task pg</code>)
      </div>
    );
  }
  if (!config) return <div className="loading">Loading…</div>;

  const voiceProfileName =
    config.voice_profiles.find((vp) => vp.id === selectedVoiceProfile)?.name ?? "";
  // Speaking is conversation state, not an audio threshold (botLevel is only a
  // waveform amplitude, never state).
  const speaking = convState === "speaking";
  const selectionName = playbookLabel(activePlaybook);
  const hasSelection = !!activePlaybook;
  const voiceLocked = appState === "connecting" || appState === "active";

  return (
    <div className={`shell shell--${appState}`}>
      <TopBar
        playbookName={selectionName}
        format={editor.format}
        appState={appState}
        hasPlaybook={hasSelection}
        dirty={editor.dirty}
        draft={editor.draft}
        valid={editor.validation.valid}
        saving={editor.saving}
        onExport={handleExport}
        onSave={editor.save}
        onPublish={editor.publish}
        onConnect={connect}
        onDisconnect={disconnect}
      />

      <div
        className="panes"
        style={{ gridTemplateColumns: `1fr 10px ${rightColWidth}px` }}
      >
        {/* LEFT — WORK PANE: preview · edit · conversation · metrics · traces · events */}
        <section className="workpane">
          <div className="center-tabs">
            {LEFT_TABS.map((t) => (
              <button
                key={t.id}
                className={`ctab${tab === t.id ? " on" : ""}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
                {t.id === "preview" && isTurnActive(convState) && (
                  <span className="tab-live" />
                )}
              </button>
            ))}
          </div>
          <div className="center-body">
            {tab === "preview" && (
              <PipelinePanel
                live={appState === "active" || appState === "connecting"}
                convState={convState}
                userText={lastUserText}
                replyText={lastAgentText}
                node={currentNode}
              />
            )}
            {tab === "edit" && (
              <EditPanel
                editor={editor}
                appState={appState}
                hasPlaybook={hasSelection}
              />
            )}
            {tab === "conversation" && (
              <ConversationPanel turns={turns} appState={appState} />
            )}
            {tab === "metrics" && <MetricsPanel metrics={metrics} />}
            {tab === "traces" && (
              <DashboardPanel
                llmCalls={llmCalls}
                turnTimings={turnTimings}
                metrics={metrics}
                turns={turns}
              />
            )}
            {tab === "events" && <EventsLog entries={events} />}
          </div>
          {/* Composer is voice-first; only the conversational tabs can "talk". */}
          {(tab === "preview" || tab === "conversation") && (
            <Composer appState={appState} speaking={speaking} />
          )}
        </section>

        <div
          className="col-resize-handle"
          onMouseDown={(e) =>
            startResize(e, "x", rightColWidth, setRightColWidth, -1, 280, 560)
          }
        />

        {/* RIGHT — CONTROL PANE: voice dropdown + Chat | Stats | Playbooks */}
        <aside className="rail right controlpane">
          <div className="controlpane-head">
            <VoiceProfilePanel
              voiceProfiles={config.voice_profiles}
              selected={selectedVoiceProfile}
              onChange={setSelectedVoiceProfile}
              disabled={voiceLocked}
            />
          </div>
          <div className="rail-tabs" role="tablist" aria-label="Control">
            {RIGHT_TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={rightPaneTab === t.id}
                className={`rail-tab${rightPaneTab === t.id ? " on" : ""}`}
                onClick={() => setRightPaneTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="controlpane-body">
            {rightPaneTab === "chat" && (
              <ChatPanel
                playbookId={activePlaybook}
                onApplied={(y) => {
                  editor.inject(y);
                  setTab("edit");
                }}
              />
            )}
            {rightPaneTab === "stats" && (
              <StatsPanel
                appState={appState}
                agentReady={agentReady}
                convState={convState}
                session={session}
                voiceProfileName={voiceProfileName}
                activeLlm={config.active_llm}
                metrics={metrics}
                botLevel={botLevel}
              />
            )}
            {rightPaneTab === "playbooks" && (
              <PlaybookList
                playbooks={playbooks}
                activePlaybook={activePlaybook}
                appState={appState}
                onSelect={selectPlaybook}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
