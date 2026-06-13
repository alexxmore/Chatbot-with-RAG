const BASE = "";

export interface Source {
  file: string;
  title: string;
  section: string;
  relevance: number;
}

export interface Usage {
  embedding_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  usage?: Usage;
}

export interface StatusResponse {
  status: "idle" | "running" | "done" | "error";
  message: string;
  embedding_tokens?: number;
}

export async function sendMessage(message: string, top_k = 5): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, top_k }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Помилка сервера");
  }
  return res.json();
}

export async function triggerReindex(): Promise<{ detail: string }> {
  const res = await fetch(`${BASE}/reindex`, { method: "POST" });
  if (!res.ok) throw new Error("Помилка запуску індексування");
  return res.json();
}

export async function getStatus(): Promise<StatusResponse> {
  const res = await fetch(`${BASE}/status`);
  if (!res.ok) throw new Error("Помилка отримання статусу");
  return res.json();
}
