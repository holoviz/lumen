from __future__ import annotations

import sqlite3
import uuid

from typing import Literal

import pandas as pd
import param

from panel import Card
from panel.chat import ChatMessage
from panel.widgets import Tabulator

from lumen.ai.utils import log_debug


class ChatLogs(param.Parameterized):
    """A class for managing and storing chat message logs in SQLite.

    The class handles storage, retrieval, and versioning of chat messages,
    including support for message retries and status tracking.
    """

    filename: str = param.String(default="lumen_chat_logs.db")
    session_id: str = param.String(constant=True)

    def __init__(self, **params) -> None:
        if "session_id" not in params:
            params["session_id"] = uuid.uuid4().hex[:8]
        super().__init__(**params)
        self.conn: sqlite3.Connection = sqlite3.connect(self.filename)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        # Create messages table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                message_index INTEGER,
                message_user TEXT,
                message_content TEXT,
                attempt_number INTEGER DEFAULT 0,
                liked BOOLEAN DEFAULT FALSE,
                disliked BOOLEAN DEFAULT FALSE,
                state TEXT DEFAULT 'active',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create retries table with foreign key constraint and attempt number
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS retries (
                retry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                message_content TEXT NOT NULL,
                liked BOOLEAN DEFAULT FALSE,
                disliked BOOLEAN DEFAULT FALSE,
                revision_type TEXT DEFAULT 'edited',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages(message_id),
                UNIQUE(message_id, attempt_number)
            )
        """)

        self.conn.commit()

    def _serialize_message(self, message: ChatMessage | str) -> str:
        """Serialize a message object into a string representation.

        Parameters
        ----------
        message : ChatMessage or str
            The message to serialize. Can be either a ChatMessage object
            or a string

        Returns
        -------
        str
            The serialized message content

        Notes
        -----
        Handles special cases for Card and Tabulator objects within
        ChatMessage objects
        """
        if isinstance(message, str):
            return message
        if isinstance(message.object, Card):
            steps = message.object
            serialized = "# Steps"
            for step in steps.objects:
                content = "\n".join([obj.object for obj in step.objects])
                serialized += f"## {step.title}\n{content}"
            return serialized
        if isinstance(message.object, Tabulator):
            tabulator = message.object
            serialized = "# Table\n"
            serialized += tabulator.value.to_markdown()
            return serialized
        return message.serialize()

    def upsert(
        self,
        message_id: str,
        message_index: int,
        message_user: str,
        message: ChatMessage | str,
    ) -> None:
        """Insert or update a message in the messages table.

        Parameters
        ----------
        message_id : str
            Unique identifier for the message
        message_index : int
            Index position of the message in the conversation
        message_user : str
            Identifier of the user who sent the message
        message : ChatMessage or str
            The message content to store

        Notes
        -----
        If a message with the same message_id exists, it will be updated
        with the new content and timestamp
        """
        content = self._serialize_message(message)
        try:
            self.cursor.execute("""
                INSERT INTO messages (
                    message_id, session_id, message_index,
                    message_user, message_content
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    message_content = excluded.message_content,
                    message_index = excluded.message_index,
                    message_user = excluded.message_user,
                    timestamp = CURRENT_TIMESTAMP
                """, (
                    message_id, self.session_id, message_index,
                    message_user, content
                )
            )
            self.conn.commit()
        except Exception as e:
            log_debug(f"Failed to upsert message: {e!s}")

    def update_retry(
        self,
        message_id: str,
        message: ChatMessage | str,
    ) -> None:
        """Create a retry version of a message.

        Parameters
        ----------
        message_id : str
            The ID of the message to retry
        message : ChatMessage or str
            The new message content

        Notes
        -----
        The method follows these steps:
        1. Checks if this is the first retry
        2. If first retry, copies original message to retries as attempt 0
        3. Creates new retry entry with incremented attempt number
        4. Updates original message state

        Raises
        ------
        ValueError
            If no original message is found for the given message_id
        """
        content = self._serialize_message(message)
        try:
            # Begin transaction
            self.cursor.execute("BEGIN")

            # Check if this is the first retry
            self.cursor.execute("""
                SELECT COUNT(*), message_content, liked, disliked
                FROM messages
                WHERE message_id = ?
                """, (message_id,))

            count, original_content, original_liked, original_disliked = self.cursor.fetchone()
            if count == 0:
                raise ValueError(f"No original message found for message_id: {message_id}")

            if content == original_content:
                log_debug("Matching content; no need to insert update")
                return

            # Get the latest attempt number
            self.cursor.execute("""
                SELECT MAX(attempt_number)
                FROM retries
                WHERE message_id = ?
                """, (message_id,))

            last_attempt = self.cursor.fetchone()[0]

            # If this is the first retry, copy original message as attempt 0
            if last_attempt is None:
                self.cursor.execute("""
                    INSERT INTO retries (
                        message_id,
                        attempt_number,
                        message_content,
                        liked,
                        disliked
                    )
                    VALUES (?, 0, ?, ?, ?)
                    """, (message_id, original_content, original_liked, original_disliked)
                )
                last_attempt = 0

            # Get status from the last attempt
            self.cursor.execute("""
                SELECT liked, disliked
                FROM retries
                WHERE message_id = ? AND attempt_number = ?
                """, (message_id, last_attempt))

            prev_liked, prev_disliked = self.cursor.fetchone()
            next_attempt = last_attempt + 1

            # Insert new retry version
            self.cursor.execute("""
                INSERT INTO retries (
                    message_id,
                    attempt_number,
                    message_content,
                    liked,
                    disliked
                )
                VALUES (?, ?, ?, ?, ?)
                """, (message_id, next_attempt, content, prev_liked, prev_disliked)
            )

            # Update original message state and contents
            self.cursor.execute("""
                UPDATE messages
                SET state = 'edited',
                    message_content = ?,
                    attempt_number = ?
                WHERE message_id = ?
                """, (content, next_attempt, message_id)
            )

            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            log_debug(f"Unexpected error creating retry version: {e!s}")

    def update_status(
        self,
        message_id: str,
        liked: bool | None = None,
        disliked: bool | None = None,
        state: Literal["active", "reran", "undone", "cleared", "edited", "retried"] | None = None
    ) -> None:
        """Update message status for the original message and its latest retry.

        Parameters
        ----------
        message_id : str
            The ID of the message to update
        liked : bool, optional
            Whether the message was liked
        disliked : bool, optional
            Whether the message was disliked
        state : str, optional
            New state for the message. Must be one of:
            "active", "reran", "undone", "cleared", "edited", "retried"
        """
        try:
            self.cursor.execute("""
                UPDATE messages
                SET liked = COALESCE(?, liked),
                    disliked = COALESCE(?, disliked),
                    state = COALESCE(?, state)
                WHERE message_id = ?
                """, (liked, disliked, state, message_id)
            )

            if state == "retried":
                # TODO; can't get over the race condition
                self.cursor.execute("""
                    UPDATE retries
                    SET liked = COALESCE(?, liked),
                        disliked = COALESCE(?, disliked),
                        revision_type = COALESCE(?, revision_type)
                    WHERE message_id = ?
                    AND attempt_number = (
                        SELECT MAX(attempt_number)
                        FROM retries
                        WHERE message_id = ?
                    )
                    """, (liked, disliked, state, message_id, message_id)
                )
            self.conn.commit()
        except Exception as e:
            log_debug(f"Failed to update status: {e!s}")

    def get_messages(self, session_id: str | None = None) -> pd.DataFrame:
        """Retrieve messages from the database.

        Parameters
        ----------
        session_id : str or None, optional
            The session ID to filter messages by. If None, returns messages
            from the last session. If "all", returns all messages

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the messages with columns:
            message_id, message_index, message_user, message_content,
            attempt_number, liked, disliked, state, timestamp
        """
        if session_id == "all":
            df = pd.read_sql("SELECT * FROM messages ORDER BY timestamp ASC", self.conn)
            return df

        if session_id is None:
            self.cursor.execute("SELECT session_id FROM messages ORDER BY timestamp LIMIT 1")
            session_id = self.cursor.fetchone()[0]

        df = pd.read_sql("""
            SELECT message_id, message_index, message_user,
                    message_content, attempt_number, liked, disliked, state, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY message_index ASC
            """, self.conn, params=(session_id,)
        )
        return df

    def get_message_history(self, message_id: str) -> pd.DataFrame:
        """Get the complete history of a message including all retries.

        Parameters
        ----------
        message_id : str
            The ID of the message to get history for

        Returns
        -------
        pandas.DataFrame
            DataFrame containing message history with columns:
            version, message_content, liked, disliked, timestamp, revision_type
        """
        try:
            query = """
                SELECT
                    attempt_number as version,
                    message_content,
                    liked,
                    disliked,
                    timestamp,
                    revision_type
                FROM retries
                WHERE message_id = ?
                ORDER BY timestamp
            """

            df_history = pd.read_sql(query, self.conn, params=(message_id,))
            return df_history

        except Exception as e:
            log_debug(f"Failed to get message history: {e!s}")
            # Return empty DataFrame with expected columns if query fails
            return pd.DataFrame(columns=[
                'version', 'message_content', 'liked',
                'disliked', 'timestamp', 'revision_type'
            ])
