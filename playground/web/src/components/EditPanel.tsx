// playground/web/src/components/EditPanel.tsx
// The Edit tab surface: the CodeMirror editor + a validation status line. All
// state + the Save/Publish actions live in the shared usePlaybookEditor hook
// (owned by AgentView, also driving the TopBar controls).
import { footerLabel } from "../config";
import type { PlaybookEditor } from "../hooks/usePlaybookEditor";
import type { AppState } from "../types";
import { CodeEditor } from "./CodeEditor";

interface EditPanelProps {
  editor: PlaybookEditor;
  appState: AppState;
  hasPlaybook: boolean;
}

export function EditPanel({ editor, appState, hasPlaybook }: EditPanelProps) {
  if (!hasPlaybook) {
    return <div className="edit-empty">Pick a playbook to edit.</div>;
  }
  return (
    <div className="editpanel">
      <div className="editpanel-bar">
        <span className={`edit-status ${editor.validation.valid ? "ok" : "err"}`}>
          {footerLabel(editor.validation)}
        </span>
        {editor.draft && <span className="edit-draft">draft</span>}
      </div>
      <CodeEditor
        value={editor.yaml}
        onChange={editor.setYaml}
        readOnly={appState === "connecting"}
      />
    </div>
  );
}
