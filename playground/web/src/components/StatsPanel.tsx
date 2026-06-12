// playground/web/src/components/StatsPanel.tsx
// Placeholder — fleshed out in Task 15 (StatusPanel + bot audio + metrics).
import type { ConvState } from "../state/convState";
import type { AppState, MetricSnapshot, SessionInfo } from "../types";

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

export function StatsPanel(_: StatsPanelProps) {
  return <div className="rail-section" style={{ padding: 16 }}>Stats…</div>;
}
