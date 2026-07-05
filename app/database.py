import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "darts.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                game_type  TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at   TEXT,
                notes      TEXT
            );

            CREATE TABLE IF NOT EXISTS session_players (
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                player_id  INTEGER NOT NULL REFERENCES players(id),
                turn_order INTEGER NOT NULL,
                final_pos  INTEGER,
                PRIMARY KEY (session_id, player_id)
            );

            CREATE TABLE IF NOT EXISTS throws (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL REFERENCES sessions(id),
                player_id       INTEGER NOT NULL REFERENCES players(id),
                round_number    INTEGER NOT NULL,
                throw_in_round  INTEGER NOT NULL,
                x               REAL,
                y               REAL,
                segment         INTEGER,
                ring            TEXT,
                score_value     INTEGER,
                timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
                target_segment  INTEGER
            );

            CREATE TABLE IF NOT EXISTS cricket_state (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                player_id  INTEGER NOT NULL REFERENCES players(id),
                round_num  INTEGER NOT NULL,
                marks_15   INTEGER NOT NULL DEFAULT 0,
                marks_16   INTEGER NOT NULL DEFAULT 0,
                marks_17   INTEGER NOT NULL DEFAULT 0,
                marks_18   INTEGER NOT NULL DEFAULT 0,
                marks_19   INTEGER NOT NULL DEFAULT 0,
                marks_20   INTEGER NOT NULL DEFAULT 0,
                marks_bull INTEGER NOT NULL DEFAULT 0,
                points     INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS score_state (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id         INTEGER NOT NULL REFERENCES sessions(id),
                player_id          INTEGER NOT NULL REFERENCES players(id),
                round_num          INTEGER NOT NULL,
                score_remaining    INTEGER NOT NULL,
                checkout_attempted INTEGER NOT NULL DEFAULT 0,
                checkout_hit       INTEGER NOT NULL DEFAULT 0
            );
        """)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_player(name: str) -> int:
    with get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO players (name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
        return row["id"]


def get_all_players() -> list:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM players ORDER BY name").fetchall()


def create_session(game_type: str, player_ids: list, notes: str = "") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (game_type, notes) VALUES (?, ?)",
            (game_type, notes),
        )
        session_id = cur.lastrowid
        for order, pid in enumerate(player_ids, start=1):
            conn.execute(
                "INSERT INTO session_players (session_id, player_id, turn_order) VALUES (?, ?, ?)",
                (session_id, pid, order),
            )
        return session_id


def end_session(session_id: int):
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )


def record_throw(
    session_id, player_id, round_number, throw_in_round,
    x, y, segment, ring, score_value, target_segment=None
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO throws
               (session_id, player_id, round_number, throw_in_round,
                x, y, segment, ring, score_value, target_segment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, player_id, round_number, throw_in_round,
             x, y, segment, ring, score_value, target_segment),
        )
        return cur.lastrowid


def get_player_throws(player_id: int, game_type: str = None) -> list:
    with get_connection() as conn:
        if game_type:
            return conn.execute(
                """SELECT t.*, s.game_type FROM throws t
                   JOIN sessions s ON t.session_id = s.id
                   WHERE t.player_id = ? AND s.game_type = ?
                   ORDER BY t.id""",
                (player_id, game_type),
            ).fetchall()
        else:
            return conn.execute(
                """SELECT t.*, s.game_type FROM throws t
                   JOIN sessions s ON t.session_id = s.id
                   WHERE t.player_id = ?
                   ORDER BY t.id""",
                (player_id,),
            ).fetchall()


def get_session_history(player_id: int) -> list:
    with get_connection() as conn:
        return conn.execute(
            """SELECT
                   s.id          AS session_id,
                   s.game_type,
                   s.started_at,
                   s.ended_at,
                   COUNT(t.id)   AS throw_count,
                   COALESCE(SUM(t.score_value), 0) AS total_score,
                   CASE WHEN COUNT(t.id) > 0
                        THEN ROUND(SUM(t.score_value) * 3.0 / COUNT(t.id), 2)
                        ELSE 0
                   END           AS three_dart_avg
               FROM sessions s
               JOIN session_players sp ON s.id = sp.session_id AND sp.player_id = ?
               LEFT JOIN throws t      ON t.session_id = s.id AND t.player_id = ?
               GROUP BY s.id
               ORDER BY s.started_at""",
            (player_id, player_id),
        ).fetchall()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")

    # Verification test
    p1 = create_player("Alice")
    p2 = create_player("Bob")
    print(f"Created players: Alice id={p1}, Bob id={p2}")

    sid = create_session("cricket", [p1, p2], notes="test session")
    print(f"Created cricket session id={sid}")

    for i in range(1, 4):
        rid = record_throw(sid, p1, round_number=1, throw_in_round=i,
                           x=10.0 * i, y=5.0 * i,
                           segment=20, ring="single", score_value=20)
        print(f"  Recorded throw {i} -> id={rid}")

    history = get_session_history(p1)
    print("\nSession history for Alice:")
    for row in history:
        print(dict(row))
