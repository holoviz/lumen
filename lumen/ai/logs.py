from __future__ import annotations

import json
import sqlite3

import param

from .utils import log_debug


class ChatLogs(param.Parameterized):

    filename = param.String(default="chat_logs.db")

    def __init__(self, **params):
        super().__init__(**params)
        self.conn = sqlite3.connect(self.filename)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                session_id TEXT,
                message_id TEXT PRIMARY KEY,
                message_index INTEGER,
                message_user TEXT,
                message_content TEXT,
                liked BOOLEAN DEFAULT FALSE,
                disliked BOOLEAN DEFAULT FALSE,
                removed BOOLEAN DEFAULT FALSE,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS explorations (
                exploration_id TEXT PRIMARY KEY,
                session_id TEXT,
                parent_id TEXT,
                position INTEGER,
                title TEXT,
                subtitle TEXT,
                spec TEXT,
                updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def upsert(
        self,
        session_id,
        message_id,
        message_index,
        message_user,
        message_content,
    ):
        UPSERT_SCHEMA = """
        INSERT INTO logs (session_id, message_id, message_index, message_user, message_content)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (message_id)
        DO UPDATE SET
        session_id = excluded.session_id,
        message_id = excluded.message_id,
        message_index = excluded.message_index,
        message_user = excluded.message_user,
        message_content = excluded.message_content
        """
        try:
            self.cursor.execute(
                UPSERT_SCHEMA,
                (
                    session_id,
                    message_id,
                    message_index,
                    message_user,
                    message_content,
                ),
            )
        except Exception:
            try:
                self.cursor.execute(
                    UPSERT_SCHEMA,
                    (
                        session_id,
                        message_id,
                        message_index,
                        message_user,
                        str(message_content),
                    ),
                )
            except Exception:
                log_debug("Failed to insert message")
                return
            self.conn.commit()

    def upsert_exploration(
        self,
        session_id,
        exploration_id,
        parent_id,
        position,
        title,
        subtitle,
        spec,
    ):
        UPSERT_EXPLORATION_SCHEMA = """
        INSERT INTO explorations (
            session_id, exploration_id, parent_id, position, title, subtitle, spec, updated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (exploration_id)
        DO UPDATE SET
        session_id = excluded.session_id,
        parent_id = excluded.parent_id,
        position = excluded.position,
        title = excluded.title,
        subtitle = excluded.subtitle,
        spec = excluded.spec,
        updated = CURRENT_TIMESTAMP
        """
        try:
            self.cursor.execute(
                UPSERT_EXPLORATION_SCHEMA,
                (
                    session_id,
                    exploration_id,
                    parent_id,
                    position,
                    title,
                    subtitle,
                    json.dumps(spec),
                ),
            )
            self.conn.commit()
        except Exception:
            log_debug("Failed to upsert exploration")

    def load_session(self, session_id):
        self.cursor.execute(
            """
            SELECT exploration_id, parent_id, position, title, subtitle, spec
            FROM explorations
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        )
        rows = self.cursor.fetchall()
        return [
            {
                "exploration_id": row[0],
                "parent_id": row[1],
                "position": row[2],
                "title": row[3],
                "subtitle": row[4],
                "spec": json.loads(row[5]),
            }
            for row in rows
        ]

    def delete_exploration(self, exploration_id):
        self.cursor.execute(
            "DELETE FROM explorations WHERE exploration_id = ?",
            (exploration_id,),
        )
        self.conn.commit()

    def delete_stale(self, ttl_seconds):
        self.cursor.execute(
            "DELETE FROM explorations WHERE updated < datetime('now', ?)",
            (f"-{ttl_seconds} seconds",),
        )
        self.conn.commit()

    def update_status(self, message_id, liked=None, disliked=None, removed=None):
        self.cursor.execute(
            """
            UPDATE logs
            SET liked = COALESCE(?, liked), disliked = COALESCE(?, disliked), removed = COALESCE(?, removed)
            WHERE message_id = ?
            """,
            (liked, disliked, removed, message_id),
        )
        self.conn.commit()
