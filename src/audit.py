"""Audit logging.

Every LLM call, retrieval, and human decision is logged to SQLite with enough
metadata to reconstruct what happened and why. This is the operational
governance layer the Konexo job description calls 'audit trails that protect
clients from liability'.
"""

from __future__ import annotations
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from config import AUDIT_DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    contract_name TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    final_status TEXT,
    final_risk_score INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- llm_call | retrieval | human_decision | error
    model TEXT,
    prompt_version TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_stage ON events(stage);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(AUDIT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


def start_run(contract_name: str) -> str:
    init_db()
    run_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO runs (run_id, contract_name, started_at) VALUES (?, ?, ?)",
            (run_id, contract_name, _now()),
        )
    return run_id


def complete_run(run_id: str, final_status: str, final_risk_score: Optional[int]) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE runs SET completed_at = ?, final_status = ?, final_risk_score = ? WHERE run_id = ?",
            (_now(), final_status, final_risk_score, run_id),
        )


def log_event(
    run_id: str,
    stage: str,
    event_type: str,
    payload: dict[str, Any],
    model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> str:
    event_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            """INSERT INTO events
               (event_id, run_id, timestamp, stage, event_type, model,
                prompt_version, input_tokens, output_tokens, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                run_id,
                _now(),
                stage,
                event_type,
                model,
                prompt_version,
                input_tokens,
                output_tokens,
                json.dumps(payload, default=str),
            ),
        )
    return event_id


def get_events_for_run(run_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp ASC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_run_summary(run_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None
