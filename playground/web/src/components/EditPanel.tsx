// playground/web/src/components/EditPanel.tsx
// The Edit tab: load a playbook's effective YAML, validate as you type
// (debounced), Save a local draft, and Publish to the canonical example.
import { useEffect, useRef, useState } from "react";

import {
  fetchSource,
  footerLabel,
  publishSource,
  saveSource,
  validateSource,
  type PlaybookValidation,
} from "../config";
import type { AppState } from "../types";
import { CodeEditor } from "./CodeEditor";

interface EditPanelProps {
  playbookId: string;
  appState: AppState;
  /** YAML proposed by the AI builder, shown for review (not yet persisted). */
  injectedYaml?: string | null;
  onInjected?: () => void;
}

const NEUTRAL: PlaybookValidation = {
  valid: true,
  errors: [],
  steps: 0,
  journey: "",
};

type SaveState = "" | "saving" | "saved" | "publishing";

export function EditPanel({
  playbookId,
  appState,
  injectedYaml,
  onInjected,
}: EditPanelProps) {
  const [yaml, setYaml] = useState("");
  const [dirty, setDirty] = useState(false);
  const [draft, setDraft] = useState(false);
  const [saving, setSaving] = useState<SaveState>("");
  const [validation, setValidation] = useState<PlaybookValidation>(NEUTRAL);
  const debounce = useRef<number | undefined>(undefined);

  // Load source whenever the selected playbook changes.
  useEffect(() => {
    if (!playbookId) return;
    let alive = true;
    fetchSource(playbookId)
      .then((s) => {
        if (!alive || !s.ok) return;
        setYaml(s.yaml);
        setDraft(s.draft);
        setDirty(false);
        setSaving("");
        setValidation({
          valid: s.valid,
          errors: s.errors,
          steps: s.steps,
          journey: s.journey,
        });
      })
      .catch(() => {
        /* leave the editor as-is on a transient fetch error */
      });
    return () => {
      alive = false;
    };
  }, [playbookId]);

  // The AI builder handed us a rewrite — show it, mark dirty, validate, clear.
  useEffect(() => {
    if (injectedYaml == null) return;
    setYaml(injectedYaml);
    setDirty(true);
    setSaving("");
    validateSource(playbookId, injectedYaml).then(setValidation).catch(() => {});
    onInjected?.();
    // playbookId is stable for the lifetime of an injection; key on the YAML.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [injectedYaml]);

  function onChange(next: string) {
    setYaml(next);
    setDirty(true);
    setSaving("");
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      validateSource(playbookId, next).then(setValidation).catch(() => {});
    }, 400);
  }

  async function onSave() {
    setSaving("saving");
    const res = await saveSource(playbookId, yaml);
    if (res.ok) {
      setDirty(false);
      setDraft(true);
      setSaving("saved");
    } else {
      setValidation((v) => ({
        ...v,
        valid: false,
        errors: res.errors ?? v.errors,
      }));
      setSaving("");
    }
  }

  async function onPublish() {
    setSaving("publishing");
    const res = await publishSource(playbookId, yaml);
    if (res.ok) {
      setDraft(false);
      setDirty(false);
      setSaving("saved");
    } else {
      setValidation((v) => ({
        ...v,
        valid: false,
        errors: res.errors ?? v.errors,
      }));
      setSaving("");
    }
  }

  if (!playbookId) {
    return <div className="edit-empty">Pick a playbook to edit.</div>;
  }

  const saveLabel =
    saving === "saving" ? "Saving…" : dirty ? "Save" : "Saved";

  return (
    <div className="editpanel">
      <div className="editpanel-bar">
        <span className={`edit-status ${validation.valid ? "ok" : "err"}`}>
          {footerLabel(validation)}
        </span>
        <span className="edit-actions">
          {draft && <span className="edit-draft">draft</span>}
          <button
            className="btn-mini"
            disabled={!dirty || saving === "saving"}
            onClick={onSave}
          >
            {saveLabel}
          </button>
          <button
            className="btn-mini publish"
            disabled={!validation.valid || saving === "publishing"}
            onClick={onPublish}
          >
            {saving === "publishing" ? "Publishing…" : "Publish"}
          </button>
        </span>
      </div>
      <CodeEditor
        value={yaml}
        onChange={onChange}
        readOnly={appState === "connecting"}
      />
    </div>
  );
}
