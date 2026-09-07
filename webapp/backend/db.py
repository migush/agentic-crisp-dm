"""SQLite storage for the maads.mirogeorgiev.eu account product.

Deliberately not an ORM for this MVP — a single file, a handful of tables, and
plain ``sqlite3`` keep the moving parts obvious. ``init_db`` is idempotent so
it can run on every process start.

The live file ``data/webapp.db`` is never deleted. Existing deployments are
migrated in place: ``users.email`` becomes ``users.username``, ``id`` is kept
so ``data/users/<id>/artifacts/`` still matches, and ``api_keys`` / ``tasks``
are left untouched (including ciphertext blobs).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "webapp.db"

# Created only when a table is missing. Never used to wipe a live database.
_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
"""

_CREATE_REST = """
CREATE TABLE IF NOT EXISTS api_keys (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    provider        TEXT NOT NULL,
    selected_model  TEXT NOT NULL,
    ciphertext_b64  TEXT NOT NULL,
    iv_b64          TEXT NOT NULL,
    kdf_salt_b64    TEXT NOT NULL,
    kdf_params_json TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(user_id, provider)
);

CREATE TABLE IF NOT EXISTS tasks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id),
    case_name         TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model_id          TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'queued',
    started_at        TEXT,
    finished_at       TEXT,
    run_artifact_path TEXT
);

CREATE TABLE IF NOT EXISTS token_spend_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       INTEGER NOT NULL REFERENCES tasks(id),
    agent         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0
);
"""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _users_column_names(conn: sqlite3.Connection) -> set[str]:
    if not _table_exists(conn, "users"):
        return set()
    return {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}


def _migrate_users_table(conn: sqlite3.Connection) -> None:
    """In-place users rebuild when the live DB still has email/password_hash.

    Drops only the old ``users`` table after copying rows into ``users_new``.
    Does not delete ``webapp.db``, ``api_keys``, ``tasks``, or artifact dirs.
    """
    cols = _users_column_names(conn)
    if "email" not in cols and "password_hash" not in cols:
        return

    username_source = "email" if "email" in cols else "username"
    previous_fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP TABLE IF EXISTS users_new")
        conn.execute(
            """
            CREATE TABLE users_new (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"INSERT INTO users_new (id, username, created_at) "
            f"SELECT id, {username_source}, created_at FROM users"
        )
        max_id = conn.execute("SELECT MAX(id) FROM users_new").fetchone()[0]
        conn.execute("DROP TABLE users")
        conn.execute("ALTER TABLE users_new RENAME TO users")
        if max_id is not None and _table_exists(conn, "sqlite_sequence"):
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('users', 'users_new')")
            conn.execute(
                "INSERT INTO sqlite_sequence (name, seq) VALUES ('users', ?)",
                (max_id,),
            )
    finally:
        conn.execute(f"PRAGMA foreign_keys = {int(previous_fk)}")


def init_db(db_path: Path | None = None) -> None:
    """Open (or create) the SQLite file and migrate ``users`` if needed.

    Never unlinks ``db_path``. A brand-new file gets the username schema; a
    live file with ``email`` / ``password_hash`` is rewritten in place.
    """
    db_path = db_path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "users"):
            conn.executescript(_CREATE_USERS)
        _migrate_users_table(conn)
        conn.executescript(_CREATE_REST)


@contextmanager
def get_conn(db_path: Path | None = None):
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
