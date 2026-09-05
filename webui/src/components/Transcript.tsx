import type { Message } from "../api/types";

export function Transcript({ messages, title }: { messages: Message[] | string | null | undefined; title?: string }) {
  if (!messages) return null;
  const list: Message[] = typeof messages === "string" ? [{ role: "text", content: messages }] : messages;
  return (
    <section className="space-y-2">
      {title && <div className="text-[11px] font-medium uppercase tracking-wide text-muted">{title}</div>}
      {list.map((message, index) => (
        <div key={index} className="overflow-hidden rounded-md bg-raised">
          <div className="flex items-center justify-between px-3 py-1.5 text-[11px] text-muted">
            <span className="font-medium text-ink2">{message.role ?? "message"}</span>
            {typeof message.content === "string" && <span className="tabular">{message.content.length} chars</span>}
          </div>
          <pre className="transcript max-h-[32rem] overflow-auto px-3 pb-3 font-mono text-[11.5px] leading-relaxed text-ink">
            {typeof message.content === "string" ? message.content : JSON.stringify(message.content ?? message, null, 2)}
          </pre>
          {message.tool_calls ? (
            <pre className="transcript px-3 pb-3 font-mono text-[11px] text-ink2">{JSON.stringify(message.tool_calls, null, 2)}</pre>
          ) : null}
        </div>
      ))}
    </section>
  );
}

export function TextBlock({ text, title, maxHeight = 240 }: { text: string | null | undefined; title?: string; maxHeight?: number }) {
  if (!text) return null;
  return (
    <section>
      {title && <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted">{title}</div>}
      <pre className="transcript overflow-auto rounded-md bg-raised px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-ink" style={{ maxHeight }}>
        {text}
      </pre>
    </section>
  );
}
