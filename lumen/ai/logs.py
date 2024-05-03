from __future__ import annotations

import sqlite3

import param


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

    def upsert(
        self,
        session_id,
        message_id,
        message_index,
        message_user,
        message_content,
    ):
        self.cursor.execute(
            """
            INSERT INTO logs (session_id, message_id, message_index, message_user, message_content)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (message_id)
            DO UPDATE SET
            session_id = excluded.session_id,
            message_id = excluded.message_id,
            message_index = excluded.message_index,
            message_user = excluded.message_user,
            message_content = excluded.message_content
            """,
            (
                session_id,
                message_id,
                message_index,
                message_user,
                message_content,
            ),
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
