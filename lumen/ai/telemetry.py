from __future__ import annotations

import json
import sqlite3

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import param

from panel import Card
from panel.chat import ChatMessage
from panel.widgets import Tabulator

from lumen.ai.utils import log_debug, normalize_dict

if TYPE_CHECKING:
    from lumen.ai.coordinator import Coordinator


class ChatTelemetry(param.Parameterized):
    """A class for managing and storing chat message logs in SQLite.

    The class handles storage, retrieval, and versioning of chat messages,
    including support for message retries and status tracking. It also tracks
    LLM configurations, agents, coordinators, and sessions.
    """

    filename: str = param.String(default="lumen_telemetry.db")

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.conn: sqlite3.Connection = sqlite3.connect(self.filename)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        # Create messages table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                message_index INTEGER,
                message_user TEXT,
                message_content TEXT,
                attempt_number INTEGER DEFAULT 0,
                liked BOOLEAN DEFAULT FALSE,
                disliked BOOLEAN DEFAULT FALSE,
                state TEXT DEFAULT 'initial',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create retries table with foreign key constraint and attempt number
        self.cursor.execute(
            """
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
        """
        )

        # Create LLM configurations table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_configs (
                llm_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                model_kwargs JSON NOT NULL,
                temperature FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mode, model_kwargs, temperature)
            )
        """
        )

        # Create agents table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                purpose TEXT,
                requires JSON DEFAULT '[]',
                provides JSON DEFAULT '[]',
                tools JSON DEFAULT '[]',
                prompts JSON NOT NULL,
                template_overrides JSON DEFAULT '{}',
                llm_id TEXT REFERENCES llm_configs(llm_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, prompts, template_overrides, llm_id)
            )
        """
        )

        # Create coordinators table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS coordinators (
                coordinator_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                agents JSON NOT NULL,
                prompts JSON NOT NULL,
                template_overrides JSON DEFAULT '{}',
                history INTEGER DEFAULT 3,
                llm_id TEXT REFERENCES llm_configs(llm_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, prompts, template_overrides, llm_id, agents)
            )
        """
        )

        # Create sessions table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                user_info JSON,
                coordinator_id TEXT REFERENCES coordinators(coordinator_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_coordinators_name ON coordinators(name)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_llm_id ON agents(llm_id)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_coordinators_llm_id ON coordinators(llm_id)"
        )

        self.conn.commit()

    def register_coordinator(
        self,
        coordinator: Coordinator,
        username: str = "anonymous",
        user_info: dict | None = None,
    ) -> str:
        """Register a coordinator and its components in the database.

        Parameters
        ----------
        coordinator : Coordinator
            The coordinator instance to register
        username : str
            Username associated with the session
        user_info : dict, optional
            Additional user information

        Returns
        -------
        str
            The session ID for the registered coordinator

        Notes
        -----
        This method will:
        1. Register the LLM configuration
        2. Register all agents
        3. Register the coordinator
        4. Create a new session
        """
        try:
            self.cursor.execute("BEGIN")

            # Register LLM config if it exists
            llm_id = None
            if coordinator.llm is not None:
                llm_config = {
                    "llm_id": coordinator.llm.hash,
                    "mode": str(coordinator.llm.mode),  # Convert Mode enum to string
                    "model_kwargs": json.dumps(coordinator.llm.model_kwargs),
                    "temperature": coordinator.llm.temperature,
                }
                self.cursor.execute(
                    """
                    INSERT OR IGNORE INTO llm_configs
                    (llm_id, mode, model_kwargs, temperature)
                    VALUES (:llm_id, :mode, :model_kwargs, :temperature)
                    """,
                    llm_config,
                )
                llm_id = coordinator.llm.hash

            # Register agents
            agent_ids = []
            for agent in coordinator.agents:
                agent_data = {
                    "agent_id": agent.hash,
                    "name": type(agent).__name__,
                    "purpose": getattr(agent, "purpose", None),
                    "requires": json.dumps(list(agent.requires)),
                    "provides": json.dumps(list(agent.provides)),
                    "tools": json.dumps([t.name for t in getattr(agent, "tools", [])]),
                    "prompts": json.dumps(normalize_dict(agent.prompts)),
                    "template_overrides": json.dumps(
                        normalize_dict(agent.template_overrides)
                    ),
                    "llm_id": llm_id,
                }
                self.cursor.execute(
                    """
                    INSERT OR IGNORE INTO agents
                    (agent_id, name, purpose, requires, provides, tools,
                     prompts, template_overrides, llm_id)
                    VALUES (
                        :agent_id, :name, :purpose, :requires, :provides, :tools,
                        :prompts, :template_overrides, :llm_id
                    )
                    """,
                    agent_data,
                )
                agent_ids.append(agent.hash)

            # Register coordinator
            coordinator_data = {
                "coordinator_id": coordinator.hash,
                "name": type(coordinator).__name__,
                "agents": json.dumps(agent_ids),
                "prompts": json.dumps(normalize_dict(coordinator.prompts)),
                "template_overrides": json.dumps(
                    normalize_dict(coordinator.template_overrides)
                ),
                "history": coordinator.history,
                "llm_id": llm_id,
            }
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO coordinators
                (coordinator_id, name, agents, prompts, template_overrides,
                 history, llm_id)
                VALUES (
                    :coordinator_id, :name, :agents, :prompts, :template_overrides,
                    :history, :llm_id
                )
                """,
                coordinator_data,
            )

            # Create new session
            session_id = coordinator.interface._session_id
            session_data = {
                "session_id": session_id,
                "username": username,
                "user_info": json.dumps(user_info) if user_info else None,
                "coordinator_id": coordinator.hash,
            }
            self.cursor.execute(
                """
                INSERT INTO sessions
                (session_id, username,
                 user_info, coordinator_id)
                VALUES (
                    :session_id, :username,
                    :user_info, :coordinator_id
                )
                """,
                session_data,
            )

            self.conn.commit()
            return session_id

        except Exception as e:
            self.conn.rollback()
            log_debug(f"Failed to register coordinator: {e!s}")
            raise

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
        session_id: str,
        message_index: int,
        message_user: str,
        message: ChatMessage | str,
    ) -> None:
        """Insert or update a message in the messages table.

        Parameters
        ----------
        message_id : str
            Unique identifier for the message
        session_id : str
            Identifier for the session to which the message belongs.
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
            self.cursor.execute(
                """
                INSERT INTO messages (
                    message_id, session_id, message_index,
                    message_user, message_content
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    message_content = excluded.message_content,
                    message_index = excluded.message_index,
                    message_user = excluded.message_user,
                    timestamp = CURRENT_TIMESTAMP
                """,
                (message_id, session_id, message_index, message_user, content),
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

        # Check if message exists and get original content before starting transaction
        self.cursor.execute(
            """
            SELECT COUNT(*), message_content, liked, disliked
            FROM messages
            WHERE message_id = ?
            """,
            (message_id,),
        )
        count, original_content, original_liked, original_disliked = (
            self.cursor.fetchone()
        )

        if count == 0:
            raise ValueError(f"No original message found for message_id: {message_id}")

        if content == original_content:
            log_debug("Matching content; no need to insert update")
            return

        try:
            # Get the latest attempt number before starting transaction
            self.cursor.execute(
                """
                SELECT MAX(attempt_number)
                FROM retries
                WHERE message_id = ?
                """,
                (message_id,),
            )
            last_attempt = self.cursor.fetchone()[0]

            # Now begin the transaction for the updates
            self.cursor.execute("BEGIN")

            # If this is the first retry, copy original message as attempt 0
            if last_attempt is None:
                self.cursor.execute(
                    """
                    INSERT INTO retries (
                        message_id,
                        attempt_number,
                        message_content,
                        liked,
                        disliked
                    )
                    VALUES (?, 0, ?, ?, ?)
                    """,
                    (message_id, original_content, original_liked, original_disliked),
                )
                last_attempt = 0

            # Get status from the last attempt
            self.cursor.execute(
                """
                SELECT liked, disliked
                FROM retries
                WHERE message_id = ? AND attempt_number = ?
                """,
                (message_id, last_attempt),
            )
            prev_liked, prev_disliked = self.cursor.fetchone()
            next_attempt = last_attempt + 1

            # Insert new retry version
            self.cursor.execute(
                """
                INSERT INTO retries (
                    message_id,
                    attempt_number,
                    message_content,
                    liked,
                    disliked
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, next_attempt, content, prev_liked, prev_disliked),
            )

            # Update original message state and contents
            self.cursor.execute(
                """
                UPDATE messages
                SET state = 'edited',
                    message_content = ?,
                    attempt_number = ?
                WHERE message_id = ?
                """,
                (content, next_attempt, message_id),
            )

            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            log_debug(f"Unexpected error creating retry version: {e!s}")
            raise

    def update_status(
        self,
        message_id: str,
        liked: bool | None = None,
        disliked: bool | None = None,
        state: (
            Literal["initial", "reran", "undone", "cleared", "edited", "retried"] | None
        ) = None,
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
            "initial", "reran", "undone", "cleared", "edited", "retried"
        """
        try:
            # Begin transaction
            self.cursor.execute("BEGIN")

            # First, get the latest attempt number to avoid race condition
            self.cursor.execute(
                """
                SELECT MAX(attempt_number)
                FROM retries
                WHERE message_id = ?
                """,
                (message_id,),
            )
            latest_attempt = self.cursor.fetchone()[0]

            # Update the original message
            self.cursor.execute(
                """
                UPDATE messages
                SET liked = COALESCE(?, liked),
                    disliked = COALESCE(?, disliked),
                    state = COALESCE(?, state)
                WHERE message_id = ?
                """,
                (liked, disliked, state, message_id),
            )

            # If there are retries, update the latest one
            if latest_attempt is not None:
                self.cursor.execute(
                    """
                    UPDATE retries
                    SET liked = COALESCE(?, liked),
                        disliked = COALESCE(?, disliked),
                        revision_type = CASE
                            WHEN ? = 'retried' THEN ?
                            ELSE revision_type
                        END
                    WHERE message_id = ? AND attempt_number = ?
                    """,
                    (liked, disliked, state, state, message_id, latest_attempt),
                )

            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
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
            self.cursor.execute(
                "SELECT session_id FROM messages ORDER BY timestamp LIMIT 1"
            )
            session_id = self.cursor.fetchone()[0]

        df = pd.read_sql(
            """
            SELECT message_id, message_index, message_user,
                    message_content, attempt_number, liked, disliked, state, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY message_index ASC
            """,
            self.conn,
            params=(session_id,),
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

    def get_sessions(
        self, session_id: str | None = "all", username: str | None = None
    ) -> pd.DataFrame:
        """Retrieve sessions from the database.

        Parameters
        ----------
        session_id : str or None, optional
            Session ID to retrieve. If "all", returns all sessions.
            If None, returns the most recent session.
        username : str or None, optional
            Username to filter sessions by. Only used if session_id is "all".

        Returns
        -------
        pandas.DataFrame
            DataFrame containing sessions with columns:
            session_id, username, coordinator_id, created_at, last_active,
            plus associated user_info if available
        """
        if session_id == "all":
            query = """
                SELECT session_id, username, coordinator_id,
                    created_at, last_active, user_info
                FROM sessions
            """
            params = []

            if username is not None:
                query += " WHERE username = ?"
                params.append(username)

            query += " ORDER BY created_at DESC"

        else:
            if session_id is None:
                # Get the most recent session
                self.cursor.execute(
                    "SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT 1"
                )
                result = self.cursor.fetchone()
                if result is None:
                    return pd.DataFrame(
                        columns=[
                            "session_id",
                            "username",
                            "coordinator_id",
                            "created_at",
                            "last_active",
                            "user_info",
                        ]
                    )
                session_id = result[0]

            query = """
                SELECT session_id, username, coordinator_id,
                    created_at, last_active, user_info
                FROM sessions
                WHERE session_id = ?
            """
            params = [session_id]

        df = pd.read_sql(query, self.conn, params=params)

        # Parse JSON user_info if present
        if "user_info" in df.columns:
            df["user_info"] = df["user_info"].apply(
                lambda x: json.loads(x) if pd.notna(x) else None
            )

        return df

    def get_coordinator(self, coordinator_id: str) -> dict[str, Any] | None:
        """Retrieve coordinator details including associated agents and LLM config.

        Parameters
        ----------
        coordinator_id : str
            ID of the coordinator to retrieve

        Returns
        -------
        dict or None
            Dictionary containing coordinator details including:
            - Basic coordinator info (name, prompts, etc.)
            - Associated agents list
            - LLM configuration
            Returns None if coordinator not found
        """
        self.cursor.execute(
            """
            SELECT c.*, l.mode, l.model_kwargs, l.temperature
            FROM coordinators c
            LEFT JOIN llm_configs l ON c.llm_id = l.llm_id
            WHERE c.coordinator_id = ?
            """,
            (coordinator_id,),
        )
        row = self.cursor.fetchone()

        if not row:
            return None

        # Convert row to dict and parse JSON fields
        coordinator_info = {
            "coordinator_id": row[0],
            "name": row[1],
            "agents": json.loads(row[2]),
            "prompts": json.loads(row[3]),
            "template_overrides": json.loads(row[4]),
            "history": row[5],
            "llm_id": row[6],
            "created_at": row[7],
            "llm_config": (
                {
                    "mode": row[8],
                    "model_kwargs": json.loads(row[9]) if row[9] else None,
                    "temperature": row[10],
                }
                if row[8]
                else None
            ),
        }

        return coordinator_info

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Retrieve agent details including associated LLM config.

        Parameters
        ----------
        agent_id : str
            ID of the agent to retrieve

        Returns
        -------
        dict or None
            Dictionary containing agent details including:
            - Basic agent info (name, purpose, requires, provides, etc.)
            - LLM configuration
            Returns None if agent not found
        """
        self.cursor.execute(
            """
            SELECT a.*, l.mode, l.model_kwargs, l.temperature
            FROM agents a
            LEFT JOIN llm_configs l ON a.llm_id = l.llm_id
            WHERE a.agent_id = ?
            """,
            (agent_id,),
        )
        row = self.cursor.fetchone()

        if not row:
            return None

        # Convert row to dict and parse JSON fields
        agent_info = {
            "agent_id": row[0],
            "name": row[1],
            "purpose": row[2],
            "requires": json.loads(row[3]),
            "provides": json.loads(row[4]),
            "tools": json.loads(row[5]),
            "prompts": json.loads(row[6]),
            "template_overrides": json.loads(row[7]),
            "llm_id": row[8],
            "created_at": row[9],
            "llm_config": (
                {
                    "mode": row[10],
                    "model_kwargs": json.loads(row[11]) if row[11] else None,
                    "temperature": row[12],
                }
                if row[10]
                else None
            ),
        }

        return agent_info

    def get_session_metrics(self, session_id: str | None = None) -> pd.DataFrame:
        """Get metrics for a specific chat session."""
        if session_id is None:
            self.cursor.execute(
                "SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT 1"
            )
            result = self.cursor.fetchone()
            if result is None:
                return pd.DataFrame()
            session_id = result[0]
        elif session_id == "all":
            return pd.concat(
                [
                    self.get_session_metrics(session_id=session_id)
                    for session_id in self.get_sessions(session_id="all")[
                        "session_id"
                    ].tolist()
                ]
            )

        # Get basic message metrics
        self.cursor.execute(
            """
            SELECT
                -- Basic counts
                SUM(CASE WHEN message_user = 'User' THEN 1 ELSE 0 END) as user_messages,
                SUM(CASE WHEN message_user != 'User' THEN 1 ELSE 0 END) as assistant_messages,
                MIN(timestamp) as session_start,
                MAX(timestamp) as session_end,
                SUM(CASE WHEN liked = TRUE THEN 1 ELSE 0 END) as total_likes,
                SUM(CASE WHEN disliked = TRUE THEN 1 ELSE 0 END) as total_dislikes,

                -- Message complexity
                AVG(CASE WHEN message_user = 'User'
                    THEN LENGTH(message_content) - LENGTH(REPLACE(message_content, ' ', '')) + 1
                    END) as avg_user_words,
                AVG(CASE WHEN message_user != 'User'
                    THEN LENGTH(message_content) - LENGTH(REPLACE(message_content, ' ', '')) + 1
                    END) as avg_assistant_words,
                AVG(LENGTH(message_content)) as avg_chars_per_message,

                -- Interaction patterns
                COUNT(DISTINCT CASE WHEN state != 'initial' THEN message_id END) as edited_messages,
                COUNT(DISTINCT CASE WHEN state = 'retried' THEN message_id END) as retried_messages,
                COUNT(DISTINCT CASE WHEN state = 'undone' THEN message_id END) as undone_messages
            FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = self.cursor.fetchone()

        # Build initial metrics dict
        metrics = {
            "user_messages": row[0],
            "assistant_messages": row[1],
            "session_duration": (
                (
                    datetime.fromisoformat(row[3]) - datetime.fromisoformat(row[2])
                ).total_seconds()
                if row[2] and row[3]
                else 0
            ),
            "total_likes": row[4],
            "total_dislikes": row[5],
            "avg_user_words": row[6],
            "avg_assistant_words": row[7],
            "avg_chars_per_message": row[8],
            "edited_messages": row[9],
            "retried_messages": row[10],
            "undone_messages": row[11],
        }

        # Get response time metrics using self-join
        self.cursor.execute(
            """
            SELECT
                -- Assistant response times
                AVG(CASE
                    WHEN m1.message_user != 'User' AND m2.message_user = 'User'
                    THEN CAST((julianday(m1.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as avg_assistant_response_s,
                MIN(CASE
                    WHEN m1.message_user != 'User' AND m2.message_user = 'User'
                    THEN CAST((julianday(m1.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as min_assistant_response_s,
                MAX(CASE
                    WHEN m1.message_user != 'User' AND m2.message_user = 'User'
                    THEN CAST((julianday(m1.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as max_assistant_response_s,

                -- User response times
                AVG(CASE
                    WHEN m1.message_user = 'User' AND m2.message_user != 'User'
                    THEN CAST((julianday(m1.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as avg_user_response_s,
                MIN(CASE
                    WHEN m1.message_user = 'User' AND m2.message_user != 'User'
                    THEN CAST((julianday(m1.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as min_user_response_s,
                MAX(CASE
                    WHEN m1.message_user = 'User' AND m2.message_user != 'User'
                    THEN CAST((julianday(m1.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as max_user_response_s
            FROM messages m1
            LEFT JOIN messages m2 ON m2.message_index = m1.message_index - 1
            AND m2.session_id = m1.session_id
            WHERE m1.session_id = ?
            """,
            (session_id,),
        )
        row = self.cursor.fetchone()

        # Add response time metrics
        metrics.update(
            {
                "avg_assistant_response_s": row[0],
                "min_assistant_response_s": row[1],
                "max_assistant_response_s": row[2],
                "avg_user_response_s": row[3],
                "min_user_response_s": row[4],
                "max_user_response_s": row[5],
            }
        )

        # Get Assistant-to-Lumen response time metrics
        self.cursor.execute(
            """
            SELECT
                AVG(CASE
                    WHEN m1.message_user = 'Lumen' AND m2.message_user = 'Assistant'
                    THEN CAST((julianday(m2.timestamp) - julianday(m1.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as avg_lumen_response_s,
                MIN(CASE
                    WHEN m1.message_user = 'Lumen' AND m2.message_user = 'Assistant'
                    THEN CAST((julianday(m2.timestamp) - julianday(m1.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as min_lumen_response_s,
                MAX(CASE
                    WHEN m1.message_user = 'Lumen' AND m2.message_user = 'Assistant'
                    THEN CAST((julianday(m2.timestamp) - julianday(m1.timestamp)) * 24 * 60 * 60 AS INTEGER)
                END) as max_lumen_response_s
            FROM messages m1
            LEFT JOIN messages m2 ON m2.message_index = m1.message_index - 1
            AND m2.session_id = m1.session_id
            WHERE m1.session_id = ?
            """,
            (session_id,),
        )
        row = self.cursor.fetchone()

        # Add Assistant-to-Lumen response time metrics
        metrics.update(
            {
                "avg_lumen_response_s": row[0],
                "min_lumen_response_s": row[1],
                "max_lumen_response_s": row[2],
            }
        )

        return pd.DataFrame([metrics], index=[session_id]).rename_axis("session_id")
