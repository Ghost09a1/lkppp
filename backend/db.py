import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def init_db(db_path: str) -> None:
    """Create tables if they do not exist and ensure expected columns."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            visual_style TEXT,
            appearance_notes TEXT,
            personality TEXT,
            backstory TEXT,
            relationship_type TEXT,
            dos TEXT,
            donts TEXT,
            voice_style TEXT DEFAULT '',
            voice_pitch_shift REAL DEFAULT 0.0,
            voice_speed REAL DEFAULT 1.0,
            voice_ref_path TEXT DEFAULT '',
            voice_youtube_url TEXT DEFAULT '',
            voice_model_path TEXT DEFAULT '',
            voice_training_status TEXT DEFAULT '',
            voice_error TEXT DEFAULT '',
            language TEXT DEFAULT 'en'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(character_id) REFERENCES characters(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(character_id) REFERENCES characters(id)
        )
        """
    )
    # ensure new columns on existing DBs
    expected_cols = {
        "voice_style": "TEXT DEFAULT ''",
        "voice_pitch_shift": "REAL DEFAULT 0.0",
        "voice_speed": "REAL DEFAULT 1.0",
        "voice_ref_path": "TEXT DEFAULT ''",
        "voice_youtube_url": "TEXT DEFAULT ''",
        "voice_model_path": "TEXT DEFAULT ''",
        "voice_training_status": "TEXT DEFAULT ''",
        "voice_error": "TEXT DEFAULT ''",
        "language": "TEXT DEFAULT 'en'",
    }
    cur.execute("PRAGMA table_info(characters)")
    existing = {row[1] for row in cur.fetchall()}
    for col, definition in expected_cols.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE characters ADD COLUMN {col} {definition}")
    conn.commit()
    conn.close()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def add_character(conn: sqlite3.Connection, payload: Dict[str, Any]) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO characters (name, description, visual_style, appearance_notes, personality, backstory, relationship_type, dos, donts,
                                voice_style, voice_pitch_shift, voice_speed, voice_ref_path, voice_youtube_url, voice_model_path,
                                voice_training_status, voice_error, language)
        VALUES (:name, :description, :visual_style, :appearance_notes, :personality, :backstory, :relationship_type, :dos, :donts,
                :voice_style, :voice_pitch_shift, :voice_speed, :voice_ref_path, :voice_youtube_url, :voice_model_path,
                :voice_training_status, :voice_error, :language)
        """,
        payload,
    )
    conn.commit()
    return cur.lastrowid


def list_characters(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM characters ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def get_character(conn: sqlite3.Connection, character_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
    return dict(row) if row else None


def store_message(conn: sqlite3.Connection, character_id: int, role: str, content: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (character_id, role, content) VALUES (?, ?, ?)",
        (character_id, role, content),
    )
    conn.commit()


def get_recent_messages(conn: sqlite3.Connection, character_id: int, limit: int) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT role, content FROM messages
        WHERE character_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (character_id, limit),
    ).fetchall()
    # Reverse to chronological
    return [(row["role"], row["content"]) for row in rows[::-1]]


def store_summary(conn: sqlite3.Connection, character_id: int, content: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO summaries (character_id, content) VALUES (?, ?)",
        (character_id, content),
    )
    conn.commit()


def get_latest_summary(conn: sqlite3.Connection, character_id: int) -> Optional[str]:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT content FROM summaries
        WHERE character_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (character_id,),
    ).fetchone()
    return row["content"] if row else None
