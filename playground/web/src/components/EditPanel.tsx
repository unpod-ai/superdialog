// playground/web/src/components/EditPanel.tsx
// Placeholder — fleshed out in Task 11 (CodeMirror load/validate/save/publish).
import type { AppState } from "../types";

interface EditPanelProps {
  playbookId: string;
  appState: AppState;
  injectedYaml?: string | null;
  onInjected?: () => void;
}

export function EditPanel({ playbookId }: EditPanelProps) {
  if (!playbookId) {
    return <div className="edit-empty">Pick a playbook to edit.</div>;
  }
  return <div className="edit-empty">Editor… ({playbookId})</div>;
}
