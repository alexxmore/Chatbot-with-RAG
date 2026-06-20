import { FormEvent, useEffect, useRef, useState } from "react";
import { sendMessage, type ChatResponse } from "../api/client";
import SourceList from "../components/SourceList";

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: ChatResponse["sources"];
  usage?: ChatResponse["usage"];
  error?: boolean;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Вітаю! Я консультант технічної підтримки. Задайте ваше запитання.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);

    try {
      const res = await sendMessage(text);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: res.answer, sources: res.sources, usage: res.usage },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: err instanceof Error ? err.message : "Невідома помилка",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-layout">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message message--${msg.role}${msg.error ? " message--error" : ""}`}>
            <div className="message-bubble">
              {msg.text.split("\n").map((line, j) => (
                <span key={j}>
                  {line}
                  {j < msg.text.split("\n").length - 1 && <br />}
                </span>
              ))}
            </div>
            {msg.sources && msg.sources.length > 0 && (
              <SourceList sources={msg.sources} />
            )}
            {msg.usage && msg.usage.total_tokens > 0 && (
              <div className="usage-line">
                Токени: {msg.usage.total_tokens} (embedding {msg.usage.embedding_tokens}, промпт{" "}
                {msg.usage.prompt_tokens}, відповідь {msg.usage.completion_tokens})
                {msg.usage.cost_usd != null && (
                  <> · Вартість: ${msg.usage.cost_usd.toFixed(6)}</>
                )}
                {msg.usage.latency_ms != null && (
                  <> · Затримка: {Math.round(msg.usage.latency_ms)} мс</>
                )}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message message--assistant">
            <div className="message-bubble typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Введіть запитання…"
          disabled={loading}
          autoFocus
        />
        <button className="chat-btn" type="submit" disabled={loading || !input.trim()}>
          Надіслати
        </button>
      </form>
    </div>
  );
}
