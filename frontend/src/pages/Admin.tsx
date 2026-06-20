import { useEffect, useState } from "react";
import {
  getLogs,
  getStatus,
  triggerReindex,
  type LogEntry,
  type StatusResponse,
} from "../api/client";

const KNOWN_LOG_KEYS = new Set(["ts", "level", "logger", "event", "request_id"]);

function logExtras(ev: LogEntry): string {
  return Object.entries(ev)
    .filter(([k]) => !KNOWN_LOG_KEYS.has(k))
    .map(([k, v]) => `${k}=${String(v)}`)
    .join("  ");
}

function logTime(ts: string): string {
  const d = new Date(ts);
  return isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
}

function levelColor(level: string): string {
  if (level === "ERROR") return "var(--red)";
  if (level === "WARNING") return "var(--yellow)";
  return "var(--muted)";
}

export default function Admin() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logLevel, setLogLevel] = useState("");
  const [logError, setLogError] = useState("");

  async function fetchStatus() {
    try {
      const s = await getStatus();
      setStatus(s);
    } catch {
      setError("Не вдалося отримати статус сервера.");
    }
  }

  async function fetchLogs() {
    try {
      const { events } = await getLogs(100, logLevel || undefined);
      setLogs(events);
      setLogError("");
    } catch (err) {
      setLogError(err instanceof Error ? err.message : "Помилка завантаження логів");
    }
  }

  useEffect(() => {
    fetchStatus();
  }, []);

  useEffect(() => {
    fetchLogs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logLevel]);

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

      <div className="admin-card logs-card">
        <div className="logs-header">
          <h2>Останні події</h2>
          <select
            className="logs-filter"
            value={logLevel}
            onChange={(e) => setLogLevel(e.target.value)}
          >
            <option value="">Усі рівні</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
          </select>
          <button className="refresh-btn logs-refresh" onClick={fetchLogs}>
            Оновити
          </button>
        </div>

        {logError && <div className="error-msg">{logError}</div>}

        {!logError && logs.length === 0 && (
          <div className="admin-desc">Подій ще немає.</div>
        )}

        {logs.length > 0 && (
          <div className="logs-table-wrap">
            <table className="logs-table">
              <thead>
                <tr>
                  <th>Час</th>
                  <th>Рівень</th>
                  <th>Подія</th>
                  <th>Деталі</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((ev, i) => (
                  <tr key={i}>
                    <td className="logs-time">{logTime(ev.ts)}</td>
                    <td style={{ color: levelColor(ev.level), fontWeight: 600 }}>
                      {ev.level}
                    </td>
                    <td className="logs-event">{ev.event}</td>
                    <td className="logs-extra">{logExtras(ev)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
