// playground/web/src/components/TopBar.tsx
import type { AppState } from "../types";
import type { PlaybookFormat, SaveState } from "../hooks/usePlaybookEditor";

interface TopBarProps {
  playbookName: string;
  format: PlaybookFormat;
  appState: AppState;
  // Editor status (from the shared usePlaybookEditor instance).
  hasPlaybook: boolean;
  dirty: boolean;
  draft: boolean;
  valid: boolean;
  saving: SaveState;
  onExport: () => void;
  onSave: () => void;
  onPublish: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

const FORMAT_LABEL: Record<PlaybookFormat, string> = {
  simple: "simple format",
  full: "full format",
  flow: "flow format",
  "": "",
};

/** The "● Saved / Unsaved / Draft" indicator mirroring the reference. */
function SavedDot({ dirty, draft }: { dirty: boolean; draft: boolean }) {
  const [cls, label] = dirty
    ? ["amber", "Unsaved"]
    : draft
      ? ["amber", "Draft"]
      : ["green", "Saved"];
  return (
    <span className="saved-ind">
      <span className={`led ${cls}`} /> {label}
    </span>
  );
}

export function TopBar({
  playbookName,
  format,
  appState,
  hasPlaybook,
  dirty,
  draft,
  valid,
  saving,
  onExport,
  onSave,
  onPublish,
  onConnect,
  onDisconnect,
}: TopBarProps) {
  const connected = appState === "active";
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">⏧</span>
        <span className="brand-name">{playbookName || "unpod playground"}</span>
        {format && <span className="format-badge">{FORMAT_LABEL[format]}</span>}
      </div>

      <div className="topbar-mid">
        {connected && (
          <div className="live-chip">
            <span className="led green" /> Live session · {playbookName}
          </div>
        )}
      </div>

      <div className="topbar-right">
        {hasPlaybook && (
          <>
            <SavedDot dirty={dirty} draft={draft} />
            <button className="btn btn--ghost" onClick={onExport}>
              Export
            </button>
            <button
              className="btn btn--ghost"
              onClick={onSave}
              disabled={!dirty || saving === "saving"}
            >
              {saving === "saving" ? "Saving…" : "Save"}
            </button>
            <button
              className="btn btn--publish"
              onClick={onPublish}
              disabled={!valid || saving === "publishing"}
            >
              {saving === "publishing" ? "Publishing…" : "Publish"}
            </button>
          </>
        )}
        {connected ? (
          <button className="btn btn--disconnect" onClick={onDisconnect}>
            Disconnect
          </button>
        ) : (
          <button
            className="btn btn--connect"
            onClick={onConnect}
            disabled={appState === "connecting"}
          >
            {appState === "connecting" ? (
              <>
                <span className="spin-mini" /> Connecting…
              </>
            ) : (
              <>
                <span className="led-dark" /> Connect
              </>
            )}
          </button>
        )}
      </div>
    </header>
  );
}
