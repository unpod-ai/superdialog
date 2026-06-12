// playground/web/src/config.ts

export interface VoiceProfile {
  id: string;
  name: string;
  stt?: string;
  tts?: string;
  description?: string;
}

export interface UnpodConfig {
  voice_profiles: VoiceProfile[];
  active_llm: string;
}

export async function fetchConfig(): Promise<UnpodConfig> {
  const resp = await fetch("/playground/config");
  if (!resp.ok) throw new Error(`config fetch failed: HTTP ${resp.status}`);
  return resp.json() as Promise<UnpodConfig>;
}

export interface Flow {
  id: string;
  label: string;
  nodes: number;
  initial_node: string;
  description: string;
}

export interface FlowsResponse {
  flows: Flow[];
  active: string | null;
}

export async function fetchFlows(): Promise<FlowsResponse> {
  const resp = await fetch("/playground/flows");
  if (!resp.ok) throw new Error(`flows fetch failed: HTTP ${resp.status}`);
  return resp.json() as Promise<FlowsResponse>;
}

/** A runnable playbook (the framework's default engine) the worker can run. */
export interface PlaybookInfo {
  id: string;
  label: string;
  goal: string;
  journeys: number;
  checkpoints: number;
  initial: string;
  description: string;
  /** True when an unpublished local draft exists for this playbook. */
  draft?: boolean;
}

export interface PlaybooksResponse {
  playbooks: PlaybookInfo[];
  active: string | null;
}

export async function fetchPlaybooks(): Promise<PlaybooksResponse> {
  const resp = await fetch("/playground/playbooks");
  if (!resp.ok) throw new Error(`playbooks fetch failed: HTTP ${resp.status}`);
  return resp.json() as Promise<PlaybooksResponse>;
}

/** Conversation engine the worker runs for a call. */
export type Mode = "flow" | "playbook";

export interface ControlResult {
  ok: boolean;
  error?: string;
}

/** Switch the live conversation's flow via the harness control plane. */
export async function switchFlow(
  flowId: string,
  preserveMemory = true,
): Promise<ControlResult> {
  let resp: Response;
  try {
    resp = await fetch("/playground/sessions/active/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "switch_flow",
        params: { flow: flowId, preserve_memory: preserveMemory },
      }),
    });
  } catch (err) {
    return { ok: false, error: (err as Error).message };
  }
  if (!resp.ok) return { ok: false, error: `HTTP ${resp.status}` };
  return resp.json() as Promise<ControlResult>;
}

// --- Playbook editing (source / validate / save / publish / AI edit) ---------

/** Validation verdict returned by the source / validate endpoints. */
export interface PlaybookValidation {
  valid: boolean;
  errors: string[];
  steps: number;
  journey: string;
}

export interface PlaybookSource extends PlaybookValidation {
  ok: boolean;
  id: string;
  yaml: string;
  draft: boolean;
  error?: string;
}

export interface SaveResult {
  ok: boolean;
  draft?: boolean;
  errors?: string[];
}

export interface EditResult extends PlaybookValidation {
  ok: boolean;
  yaml: string;
  summary: string;
  error?: string;
}

/** Human-readable footer for the editor: "Valid · N steps · journey: X". */
export function footerLabel(v: PlaybookValidation): string {
  if (v.valid) return `Valid · ${v.steps} steps · journey: ${v.journey}`;
  return `Invalid · ${v.errors[0] ?? "see editor"}`;
}

/** Load a playbook's effective YAML (draft if present, else canonical). */
export async function fetchSource(id: string): Promise<PlaybookSource> {
  const resp = await fetch(`/playground/playbooks/${id}/source`);
  if (!resp.ok) throw new Error(`source fetch failed: HTTP ${resp.status}`);
  return resp.json() as Promise<PlaybookSource>;
}

/** Validate candidate YAML without saving (drives the editor footer). */
export async function validateSource(
  id: string,
  yaml: string,
): Promise<PlaybookValidation> {
  const resp = await fetch(`/playground/playbooks/${id}/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml }),
  });
  return resp.json() as Promise<PlaybookValidation>;
}

/** Validate, then save the YAML as a local draft (rejected if invalid). */
export async function saveSource(id: string, yaml: string): Promise<SaveResult> {
  const resp = await fetch(`/playground/playbooks/${id}/source`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml }),
  });
  return resp.json() as Promise<SaveResult>;
}

/** Promote the draft (or a supplied YAML) to the canonical example. */
export async function publishSource(
  id: string,
  yaml?: string,
): Promise<SaveResult> {
  const resp = await fetch(`/playground/playbooks/${id}/publish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(yaml === undefined ? {} : { yaml }),
  });
  return resp.json() as Promise<SaveResult>;
}

/** Ask the AI builder to rewrite the playbook from an instruction. */
export async function editPlaybook(
  id: string,
  instruction: string,
  yaml?: string,
): Promise<EditResult> {
  const resp = await fetch(`/playground/playbooks/${id}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      yaml === undefined ? { instruction } : { instruction, yaml },
    ),
  });
  return resp.json() as Promise<EditResult>;
}