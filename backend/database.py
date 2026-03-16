# ============================================================
# DATABASE.PY
# Async SQLite interface using aiosqlite.
# ============================================================

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import aiosqlite

from backend.actions import N_ACTIONS


# ----------------------------------------------------------
# DATABASE PATH
# ----------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH: str = os.path.join(_PROJECT_ROOT, "database", "sim.db")


# ----------------------------------------------------------
# TABLE SCHEMAS
# ----------------------------------------------------------

_CREATE_ROUNDS_TABLE = """
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    atk_action_id INTEGER NOT NULL,
    atk_action_name TEXT NOT NULL,
    def_action_id INTEGER NOT NULL,
    def_action_name TEXT NOT NULL,
    blocked INTEGER NOT NULL,
    damage INTEGER NOT NULL,
    system_health INTEGER NOT NULL,
    atk_reward INTEGER NOT NULL,
    def_reward INTEGER NOT NULL,
    epsilon REAL NOT NULL,
    timestamp TEXT NOT NULL
);
"""

_CREATE_ROUNDS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_rounds_session
ON rounds (session_id, round);
"""

_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    total_rounds INTEGER NOT NULL DEFAULT 0,
    total_breaches INTEGER NOT NULL DEFAULT 0,
    total_blocks INTEGER NOT NULL DEFAULT 0,
    system_health INTEGER NOT NULL DEFAULT 100,
    is_game_over INTEGER NOT NULL DEFAULT 0,
    epsilon REAL NOT NULL DEFAULT 0.9,
    current_state INTEGER NOT NULL DEFAULT 0,
    atk_qtable TEXT NOT NULL,
    def_qtable TEXT NOT NULL,
    atk_counts TEXT NOT NULL,
    def_counts TEXT NOT NULL,
    trend_data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


# ----------------------------------------------------------
# STARTUP
# ----------------------------------------------------------

def ensure_database_dir() -> None:
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)


def init_database_sync() -> None:
    ensure_database_dir()

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(_CREATE_ROUNDS_TABLE)
        cursor.execute(_CREATE_ROUNDS_INDEX)
        cursor.execute(_CREATE_SESSIONS_TABLE)
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------
# CONNECTION MANAGER
# ----------------------------------------------------------

@asynccontextmanager
async def get_connection():
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA foreign_keys=ON;")

    try:
        yield conn
    finally:
        await conn.close()


# ----------------------------------------------------------
# WRITE: SAVE ROUND
# ----------------------------------------------------------

async def save_round(
    session_id: str,
    round_num: int,
    atk_action_id: int,
    atk_action_name: str,
    def_action_id: int,
    def_action_name: str,
    blocked: bool,
    damage: int,
    system_health: int,
    atk_reward: int,
    def_reward: int,
    epsilon: float,
) -> int:

    timestamp = datetime.now(timezone.utc).isoformat()

    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO rounds (
                session_id, round,
                atk_action_id, atk_action_name,
                def_action_id, def_action_name,
                blocked, damage, system_health,
                atk_reward, def_reward,
                epsilon, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                round_num,
                atk_action_id,
                atk_action_name,
                def_action_id,
                def_action_name,
                1 if blocked else 0,
                damage,
                system_health,
                atk_reward,
                def_reward,
                epsilon,
                timestamp,
            ),
        )

        await conn.commit()
        return cursor.lastrowid


# ----------------------------------------------------------
# SESSION UPSERT
# ----------------------------------------------------------

async def save_session(
    session_id: str,
    total_rounds: int,
    total_breaches: int,
    total_blocks: int,
    system_health: int,
    is_game_over: bool,
    epsilon: float,
    current_state: int,
    atk_qtable: list[list[float]],
    def_qtable: list[list[float]],
    atk_counts: list[int],
    def_counts: list[int],
    trend_data: list[float],
    created_at: Optional[str] = None,
) -> None:

    now = datetime.now(timezone.utc).isoformat()
    created = created_at or now

    async with get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (
                session_id,
                total_rounds, total_breaches, total_blocks,
                system_health, is_game_over,
                epsilon, current_state,
                atk_qtable, def_qtable,
                atk_counts, def_counts,
                trend_data,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                total_rounds   = excluded.total_rounds,
                total_breaches = excluded.total_breaches,
                total_blocks   = excluded.total_blocks,
                system_health  = excluded.system_health,
                is_game_over   = excluded.is_game_over,
                epsilon        = excluded.epsilon,
                current_state  = excluded.current_state,
                atk_qtable     = excluded.atk_qtable,
                def_qtable     = excluded.def_qtable,
                atk_counts     = excluded.atk_counts,
                def_counts     = excluded.def_counts,
                trend_data     = excluded.trend_data,
                updated_at     = excluded.updated_at
            """,
            (
                session_id,
                total_rounds,
                total_breaches,
                total_blocks,
                system_health,
                1 if is_game_over else 0,
                epsilon,
                current_state,
                json.dumps(atk_qtable),
                json.dumps(def_qtable),
                json.dumps(atk_counts),
                json.dumps(def_counts),
                json.dumps(trend_data),
                created,
                now,
            ),
        )

        await conn.commit()


# ----------------------------------------------------------
# LOAD SESSION
# ----------------------------------------------------------

async def load_session(session_id: str) -> Optional[dict]:

    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        return None

    return {
        "session_id": row["session_id"],
        "total_rounds": row["total_rounds"],
        "total_breaches": row["total_breaches"],
        "total_blocks": row["total_blocks"],
        "system_health": row["system_health"],
        "is_game_over": bool(row["is_game_over"]),
        "epsilon": row["epsilon"],
        "current_state": row["current_state"],
        "atk_qtable": json.loads(row["atk_qtable"]),
        "def_qtable": json.loads(row["def_qtable"]),
        "atk_counts": json.loads(row["atk_counts"]),
        "def_counts": json.loads(row["def_counts"]),
        "trend_data": json.loads(row["trend_data"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ----------------------------------------------------------
# LOAD HISTORY
# ----------------------------------------------------------

async def load_history(session_id: str, limit: int = 100, offset: int = 0):

    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT *
            FROM rounds
            WHERE session_id = ?
            ORDER BY round DESC
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        )

        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


# ----------------------------------------------------------
# LATEST SESSION
# ----------------------------------------------------------
async def count_rounds(session_id: str) -> int:
    """
    Return the total number of rounds stored for a session.
    """

    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE session_id = ?",
            (session_id,),
        )

        row = await cursor.fetchone()

    return row[0] if row else 0

async def get_latest_session_id() -> Optional[str]:

    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT session_id
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT 1
            """
        )

        row = await cursor.fetchone()

    return row["session_id"] if row else None


# ----------------------------------------------------------
# CLEAR SESSION
# ----------------------------------------------------------

async def clear_session_rounds(session_id: str) -> int:

    async with get_connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM rounds WHERE session_id = ?",
            (session_id,),
        )

        await conn.commit()
        return cursor.rowcount


# ----------------------------------------------------------
# QTABLE HELPERS
# ----------------------------------------------------------

def serialize_qtable(qtable: list[list[float]]) -> str:
    return json.dumps(qtable)


def deserialize_qtable(raw: str) -> list[list[float]]:

    qtable = json.loads(raw)

    if len(qtable) != N_ACTIONS:
        raise ValueError("Invalid Q-table shape")

    for row in qtable:
        if len(row) != N_ACTIONS:
            raise ValueError("Invalid Q-table shape")

    return qtable