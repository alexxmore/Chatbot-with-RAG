"""Central logging setup: JSON-lines to console + a rotating file.

One logger tree under "rag". Every record automatically carries the current
`request_id` (set per request by the HTTP middleware) so access logs and
application/security logs for the same request correlate.

Privacy: this module never logs message/answer text itself — callers decide
what to include, and full prompts are only logged when settings.LOG_PROMPTS is on.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings

# Set per-request by the HTTP middleware; read by the formatter below.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_LOGGER_NAME = "rag"
_configured = False


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object (one line = one event)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the 'rag' logger tree (console + rotating file). Idempotent."""
    global _configured
    if _configured:
        return

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(settings.LOG_LEVEL.upper())
    logger.propagate = False  # don't double-log through the root logger

    fmt = JsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Emit a structured event: `event` is the message, `fields` become JSON keys."""
    logger.log(level, event, extra={"fields": fields})
