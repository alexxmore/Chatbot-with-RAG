import { useEffect, useState } from "react";
import { getStatus, triggerReindex, type StatusResponse } from "../api/client";

export default function Admin() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchStatus() {
    try {
      const s = await getStatus();
      setStatus(s);
    } catch {
      setError("Не вдалося отримати статус сервера.");
    }
  }

  useEffect(() => {
    fetchStatus();
  }, []);

  // Poll while indexing is running
  useEffect(() => {
    if (status?.status !== "running") return;
    const id = setInterval(fetchStatus, 2000);
    return () => clearInterval(id);
  }, [status?.status]);

  async function handleReindex() {
    setError("");
    setLoading(true);
    try {
      await triggerReindex();
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Помилка");
    } finally {
      setLoading(false);
    }
  }

  const statusColor =
    status?.status === "done"
      ? "var(--green)"
      : status?.status === "error"
      ? "var(--red)"
      : status?.status === "running"
      ? "var(--yellow)"
      : "var(--muted)";

  return (
    <div className="admin-layout">
      <div className="admin-card">
        <h2>Управління базою знань</h2>
        <p className="admin-desc">
          Натисніть кнопку нижче, щоб проіндексувати нові або змінені файли з директорії
          інструкцій. Вже проіндексовані файли без змін залишаться в базі.
        </p>

        <div className="status-box" style={{ borderColor: statusColor }}>
          <div className="status-label">
            Статус:{" "}
            <strong style={{ color: statusColor }}>
              {status ? status.status : "…"}
            </strong>
          </div>
          {status?.message && (
            <div className="status-message">{status.message}</div>
          )}
          {status?.embedding_tokens != null && status.embedding_tokens > 0 && (
            <div className="status-tokens">
              Токени embedding (останній запуск): <strong>{status.embedding_tokens}</strong>
              {status.embedding_cost_usd != null && (
                <> · вартість <strong>${status.embedding_cost_usd.toFixed(4)}</strong></>
              )}
            </div>
          )}
        </div>

        {error && <div className="error-msg">{error}</div>}

        <button
          className="reindex-btn"
          onClick={handleReindex}
          disabled={loading || status?.status === "running"}
        >
          {status?.status === "running"
            ? "Індексування…"
            : "Оновити базу знань"}
        </button>

        <button className="refresh-btn" onClick={fetchStatus}>
          Оновити статус
        </button>
      </div>
    </div>
  );
}
