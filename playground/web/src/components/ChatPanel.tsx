// playground/web/src/components/ChatPanel.tsx
// Placeholder — fleshed out in Task 12 (AI builder: chips, composer, /edit).

interface ChatPanelProps {
  playbookId: string;
  /** Called with the rewritten YAML after a valid AI edit. */
  onApplied: (yaml: string) => void;
}

export function ChatPanel({ playbookId }: ChatPanelProps) {
  return (
    <div className="chat-hello" style={{ padding: 16 }}>
      Chat… ({playbookId || "no playbook"})
    </div>
  );
}
