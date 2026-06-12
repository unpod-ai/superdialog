// playground/web/src/components/ChatPanel.tsx
// The Chat tab: an AI builder that rewrites the playbook YAML from a
// natural-language instruction (or a suggestion chip). A valid rewrite is
// handed up via onApplied → the parent shows it in the Edit tab for review.
import { useEffect, useRef, useState } from "react";

import { editPlaybook } from "../config";

interface ChatMsg {
  role: "user" | "agent";
  text: string;
}

const CHIPS = [
  "Add an SMS confirmation step",
  "Make the agent warmer",
  "Add a callback step",
  "Add a second language",
];

interface ChatPanelProps {
  playbookId: string;
  /** Called with the rewritten YAML after a valid AI edit. */
  onApplied: (yaml: string) => void;
}

export function ChatPanel({ playbookId, onApplied }: ChatPanelProps) {
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [msgs, busy]);

  async function send(text: string) {
    const instruction = text.trim();
    if (!instruction || busy || !playbookId) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text: instruction }]);
    setBusy(true);
    let res: Awaited<ReturnType<typeof editPlaybook>> | null = null;
    try {
      res = await editPlaybook(playbookId, instruction);
    } catch (err) {
      setMsgs((m) => [
        ...m,
        { role: "agent", text: `Couldn't reach the builder: ${(err as Error).message}` },
      ]);
      setBusy(false);
      return;
    }
    setBusy(false);
    if (!res.ok) {
      setMsgs((m) => [
        ...m,
        { role: "agent", text: `Couldn't do that: ${res?.error ?? "error"}` },
      ]);
      return;
    }
    if (!res.valid) {
      setMsgs((m) => [
        ...m,
        {
          role: "agent",
          text: `That produced invalid YAML, so I kept your version. (${
            res?.errors[0] ?? ""
          })`,
        },
      ]);
      return;
    }
    setMsgs((m) => [...m, { role: "agent", text: res!.summary }]);
    onApplied(res.yaml);
  }

  return (
    <div className="chatpanel">
      <div className="chat-head">
        <span className="chat-title">Agent</span>
        <span className="chat-sub">building your playbook</span>
      </div>
      <div className="chat-thread">
        {msgs.length === 0 && (
          <p className="chat-hello">
            Tell me what to change, or pick a suggestion below. I'll edit the
            plan on the left.
          </p>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`chat-msg chat-msg--${m.role}`}>
            {m.text}
          </div>
        ))}
        {busy && (
          <div className="chat-msg chat-msg--agent chat-typing">…</div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-chips">
        {CHIPS.map((c) => (
          <button
            key={c}
            type="button"
            className="chat-chip"
            disabled={busy || !playbookId}
            onClick={() => void send(c)}
          >
            {c}
          </button>
        ))}
      </div>
      <form
        className="chat-composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the agent to change something…"
          disabled={busy || !playbookId}
          aria-label="Ask the agent to change something"
        />
        <button type="submit" disabled={busy || !input.trim()} aria-label="Send">
          ↑
        </button>
      </form>
    </div>
  );
}
