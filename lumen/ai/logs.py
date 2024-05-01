from __future__ import annotations

import datetime
import sqlite3
import uuid

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
                user TEXT,
                content TEXT,
                liked BOOLEAN DEFAULT FALSE,
                disliked BOOLEAN DEFAULT FALSE,
                removed BOOLEAN DEFAULT FALSE,
                version INTEGER DEFAULT 1,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def append(self, session_id, message_id, message_index, user, content, version=1):
        self.cursor.execute(
            """
            INSERT INTO logs (session_id, message_id, message_index, user, content, version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, message_id, message_index, user, content, version),
        )
        self.conn.commit()

    def update(self, id, liked=None, disliked=None, removed=None):
        self.cursor.execute(
            """
            UPDATE logs
            SET liked = COALESCE(?, liked), disliked = COALESCE(?, disliked), removed = COALESCE(?, removed)
            WHERE id = ?
            """,
            (liked, disliked, removed, id),
        )
        self.conn.commit()

    def upsert(self, session_id, message_id, message_index, user, content, version=1):
        self.cursor.execute(
            """
            INSERT INTO logs (session_id, message_id, message_index, user, content, version)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (message_id)
            DO UPDATE SET
            session_id = excluded.session_id,
            message_index = excluded.message_index,
            user = excluded.user,
            content = excluded.content,
            version = excluded.version
            """,
            (session_id, message_id, message_index, user, content, version),
        )
        self.conn.commit()
