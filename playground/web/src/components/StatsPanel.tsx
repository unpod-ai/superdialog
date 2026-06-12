// playground/web/src/components/StatsPanel.tsx
// The Stats tab: the live session state (StatusPanel), the bot-audio waveform,
// and the call metrics — the old left rail, minus the voice dropdown (now
// pinned above the control-pane tabs).
import type { ConvState } from "../state/convState";
import type { AppState, MetricSnapshot, SessionInfo } from "../types";
import { StatusPanel } from "./StatusPanel";
import { BotAudioPanel } from "./BotAudioPanel";
import { MetricsPanel } from "./MetricsPanel";

export interface StatsPanelProps {
  appState: AppState;
  agentReady: boolean;
  convState: ConvState;
  session: SessionInfo | null;
  voiceProfileName: string;
  activeLlm: string;
  metrics: MetricSnapshot;
  botLevel: number;
}

export function StatsPanel({
  appState,
  agentReady,
  convState,
  session,
  voiceProfileName,
  activeLlm,
  metrics,
  botLevel,
}: StatsPanelProps) {
  return (
    <div className="statspanel">
      <StatusPanel
        appState={appState}
        agentReady={agentReady}
        convState={convState}
        session={session}
        voiceProfileName={voiceProfileName}
        activeLlm={activeLlm}
        latencyMs={metrics.ttfa_ms}
      />
      <div className="rail-section">
        <div className="rail-label">Bot audio</div>
        <BotAudioPanel appState={appState} level={botLevel} />
      </div>
      <div className="rail-section">
        <div className="rail-label">Metrics</div>
        <MetricsPanel metrics={metrics} />
      </div>
    </div>
  );
}
