// playground/web/src/hooks/usePlaybookEditor.ts
// Single source of truth for the playbook editor: the YAML buffer, its
// validation verdict, draft/dirty/saving status, and the save/publish actions.
// AgentView owns one instance and shares it with both EditPanel (the CodeMirror
// surface + footer) and TopBar (the Saved · Export · Publish controls), so the
// two never diverge.
import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchSource,
  publishSource,
  saveSource,
  validateSource,
  type PlaybookValidation,
} from "../config";

export type SaveState = "" | "saving" | "saved" | "publishing";
export type PlaybookFormat = "simple" | "full" | "flow" | "";

export interface PlaybookEditor {
  yaml: string;
  /** User edit — sets dirty + debounced validate. */
  setYaml: (next: string) => void;
  /** AI edit — shows the rewrite, marks dirty, validates immediately. */
  inject: (yaml: string) => void;
  validation: PlaybookValidation;
  dirty: boolean;
  draft: boolean;
  saving: SaveState;
  format: PlaybookFormat;
  ready: boolean;
  save: () => Promise<void>;
  publish: () => Promise<void>;
}

const NEUTRAL: PlaybookValidation = {
  valid: true,
  errors: [],
  steps: 0,
  journey: "",
};

/** Cheap client-side format guess for the TopBar badge. */
function guessFormat(yaml: string): PlaybookFormat {
  if (/^\s*journeys\s*:/m.test(yaml)) return "full";
  if (/^\s*initial_node\s*:/m.test(yaml) || /^\s*nodes\s*:/m.test(yaml)) {
    return "flow";
  }
  if (/^\s*playbook\s*:/m.test(yaml)) return "simple";
  return "";
}

export function usePlaybookEditor(playbookId: string): PlaybookEditor {
  const [yaml, setYamlState] = useState("");
  const [dirty, setDirty] = useState(false);
  const [draft, setDraft] = useState(false);
  const [saving, setSaving] = useState<SaveState>("");
  const [validation, setValidation] = useState<PlaybookValidation>(NEUTRAL);
  const [ready, setReady] = useState(false);
  const debounce = useRef<number | undefined>(undefined);

  // Load source whenever the selected playbook changes.
  useEffect(() => {
    if (!playbookId) {
      setYamlState("");
      setReady(false);
      return;
    }
    let alive = true;
    setReady(false);
    fetchSource(playbookId)
      .then((s) => {
        if (!alive || !s.ok) return;
        setYamlState(s.yaml);
        setDraft(s.draft);
        setDirty(false);
        setSaving("");
        setValidation({
          valid: s.valid,
          errors: s.errors,
          steps: s.steps,
          journey: s.journey,
        });
        setReady(true);
      })
      .catch(() => {
        /* leave the buffer as-is on a transient fetch error */
      });
    return () => {
      alive = false;
    };
  }, [playbookId]);

  const setYaml = useCallback(
    (next: string) => {
      setYamlState(next);
      setDirty(true);
      setSaving("");
      window.clearTimeout(debounce.current);
      debounce.current = window.setTimeout(() => {
        validateSource(playbookId, next).then(setValidation).catch(() => {});
      }, 400);
    },
    [playbookId],
  );

  const inject = useCallback(
    (next: string) => {
      setYamlState(next);
      setDirty(true);
      setSaving("");
      validateSource(playbookId, next).then(setValidation).catch(() => {});
    },
    [playbookId],
  );

  const save = useCallback(async () => {
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
  }, [playbookId, yaml]);

  const publish = useCallback(async () => {
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
  }, [playbookId, yaml]);

  return {
    yaml,
    setYaml,
    inject,
    validation,
    dirty,
    draft,
    saving,
    format: guessFormat(yaml),
    ready,
    save,
    publish,
  };
}
