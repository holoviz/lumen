from __future__ import annotations

import json
import sqlite3

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import param

from panel import Card
from panel.chat import ChatMessage
from panel.widgets import Tabulator

from lumen.ai.utils import log_debug, normalize_value

try:
    from .views import LumenOutput
except ImportError:
    # For testing where views might not be available
    LumenOutput = None

if TYPE_CHECKING:
    from lumen.ai.coordinator import Coordinator

    from .views import LumenOutput


# Schema version allows tracking changes and implementing migrations
SCHEMA_VERSION = "1.0.0"
PPRINT_WIDTH = 360


@dataclass
class LLMConfig:
    """Configuration for a language model."""

    llm_id: str
    mode: str
    model_kwargs: dict[str, Any]
    temperature: float
    created_at: datetime

    def pprint(self):
        pprint(self.__dict__, width=PPRINT_WIDTH)


@dataclass
class AgentConfig:
    """Configuration for an agent in the system."""

    agent_id: str
    name: str
    purpose: str | None
    requires: list[str]
    provides: list[str]
    tools: list[str]
    prompts: dict[str, Any]
    template_overrides: dict[str, Any]
    llm: LLMConfig | None
    created_at: datetime

    def pprint(self):
        pprint(self.__dict__, width=PPRINT_WIDTH)


@dataclass
class CoordinatorConfig:
    """Configuration for a coordinator managing multiple agents."""

    coordinator_id: str
    name: str
    agents: list[AgentConfig]
    prompts: dict[str, Any]
    template_overrides: dict[str, Any]
    tools: list[str]
    history: int
    llm: LLMConfig | None
    created_at: datetime

    def pprint(self):
        pprint(self.__dict__, width=PPRINT_WIDTH)


@dataclass
class SessionInfo:
    """Information about a chat session."""

    session_id: str
    username: str
    user_info: dict[str, Any] | None
    coordinator: CoordinatorConfig
    created_at: datetime
    last_active: datetime
    metrics: dict

    def pprint(self):
        pprint(self.__dict__, width=PPRINT_WIDTH)


class ChatLogs(param.Parameterized):
    """Abstract base class for chat logging implementations.

    This class defines the interface for all chat logging implementations.
    Concrete subclasses must implement these methods to provide specific
    persistence mechanisms.
    """

    def _serialize_message(self, message: ChatMessage | str) -> dict:
        """Serialize a message object into a structured dictionary representation.

        Parameters
        ----------
        message : ChatMessage or str
            The message to serialize. Can be either a ChatMessage object
            or a string

        Returns
        -------
        dict
            The serialized message content as a structured dictionary

        Notes
        -----
        This method handles various message types and objects within ChatMessage objects,
        converting them to a structured format that can be easily stored and restored.
        """
        if isinstance(message, str):
            return {"type": "text", "content": message}

        if isinstance(message.object, Card):
            steps = []
            for step in message.object.objects:
                if hasattr(step, 'objects'):
                    content = "\n".join([str(obj.object) for obj in step.objects])
                else:
                    content = str(step.object) if hasattr(step, 'object') else ""

                step_dict = {
                    "title": step.title if hasattr(step, 'title') else "",
                    "content": content,
                }

                if hasattr(step, 'status'):
                    step_dict["status"] = step.status
                if hasattr(step, 'success_title'):
                    step_dict["success_title"] = step.success_title
                if hasattr(step, 'failed_title'):
                    step_dict["failed_title"] = step.failed_title
                steps.append(step_dict)

            return {
                "type": "card",
                "steps": steps,
                "title": message.object.title if hasattr(message.object, 'title') else ""
            }

        elif isinstance(message.object, LumenOutput):
            output = message.object
            output_data = {
                "type": "lumen_output",
                "output_type": output.__class__.__name__,
                "title": output.title,
                "spec": output.spec,
                "language": output.language,
                "active": output.active
            }

            if output.__class__.__name__ == "AnalysisOutput":
                # TODO: double check if this is the right way to do it
                output_data["analysis_type"] = output.analysis.to_spec()
                output_data["pipeline"] = output.pipeline.to_spec()
            return output_data

        elif isinstance(message.object, Tabulator):
            data = message.object.value.to_dict(orient="records")
            return {
                "type": "table",
                "data": data,
            }

        return {
            "type": "message",
            "content": message.serialize(),
            "user": message.user,
            "name": message.name if hasattr(message, 'name') else None
        }

    def _serialize_message_to_string(self, message: ChatMessage | str | dict) -> str:
        """Convert a serialized message dictionary back to a string representation.

        This is used for backward compatibility and for displaying messages in text format.

        Parameters
        ----------
        message : ChatMessage or str or dict
            The message to serialize to string

        Returns
        -------
        str
            String representation of the message
        """
        # If already a string, return as is
        if isinstance(message, str):
            return message

        # If the message is a ChatMessage, serialize it first
        if isinstance(message, ChatMessage):
            message_dict = self._serialize_message(message)
        elif isinstance(message, dict):
            message_dict = message
        else:
            return str(message)

        # Convert dict to string based on type
        msg_type = message_dict.get("type", "message")

        if msg_type == "text":
            return message_dict["content"]

        elif msg_type == "card":
            serialized = "# Steps"
            for step in message_dict.get("steps", []):
                title = step.get("title", "")
                content = step.get("content", "")
                serialized += f"## {title}\n{content}"
            return serialized

        elif msg_type == "lumen_output":
            output_type = message_dict.get("output_type", "LumenOutput")
            title = message_dict.get("title", "")
            spec = message_dict.get("spec", "")

            if output_type == "SQLOutput":
                return f"SQLOutput:\n```sql\n{spec}\n```"
            elif "analysis_type" in message_dict:
                return f"{output_type} ({message_dict['analysis_type']}):\n```yaml\n{spec}\n```"
            else:
                return f"{output_type}:\n```yaml\n{spec}\n```"

        elif msg_type == "table":
            return message_dict.get("markdown", "# Table\n(Table data not available)")

        elif msg_type == "message":
            return message_dict.get("content", "")

        # Fallback
        return json.dumps(message_dict)

    @abstractmethod
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
        """

    @abstractmethod
    def upsert(
        self,
        message_id: str,
        session_id: str,
        message_index: int,
        message_user: str,
        message: ChatMessage | str,
        memory: dict | None = None,
    ) -> None:
        """Insert or update a message.

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
        memory : dict, optional
            Memory state at the time of the message
        """

    @abstractmethod
    def update_retry(
        self,
        message_id: str,
        message: ChatMessage | str,
        memory: dict | None = None,
    ) -> None:
        """Create a retry version of a message.

        Parameters
        ----------
        message_id : str
            The ID of the message to retry
        message : ChatMessage or str
            The new message content
        memory : dict, optional
            Memory state at the time of the retry
        """

    @abstractmethod
    def update_status(
        self,
        message_id: str,
        liked: bool | None = None,
        disliked: bool | None = None,
        state: (
            Literal["initial", "reran", "undone", "cleared", "edited", "retried"] | None
        ) = None,
    ) -> None:
        """Update message status.

        Parameters
        ----------
        message_id : str
            The ID of the message to update
        liked : bool, optional
            Whether the message was liked
        disliked : bool, optional
            Whether the message was disliked
        state : str, optional
            New state for the message
        """

    @abstractmethod
    def view_messages(self, session_id: str | None = None) -> pd.DataFrame:
        """Retrieve messages.

        Parameters
        ----------
        session_id : str or None, optional
            The session ID to filter messages by

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the messages
        """

    @abstractmethod
    def view_message_history(self, message_id: str) -> pd.DataFrame:
        """Get the complete history of a message including all retries.

        Parameters
        ----------
        message_id : str
            The ID of the message to get history for

        Returns
        -------
        pandas.DataFrame
            DataFrame containing message history
        """

    @abstractmethod
    def view_sessions(
        self, session_id: str | None = "all", username: str | None = None
    ) -> pd.DataFrame:
        """View sessions with their associated metrics.

        Parameters
        ----------
        session_id : str or None, optional
            Session ID to retrieve
        username : str or None, optional
            Username to filter sessions by

        Returns
        -------
        pandas.DataFrame
            DataFrame containing sessions
        """

    @abstractmethod
    def view_coordinator_agents(
        self, coordinator_id: str | None = None
    ) -> pd.DataFrame:
        """View detailed information about all agents in a coordinator.

        Parameters
        ----------
        coordinator_id : str or None, optional
            The coordinator ID to retrieve information for

        Returns
        -------
        pandas.DataFrame
            DataFrame containing detailed agent information
        """

    @abstractmethod
    def get_llm_config(self, llm_id: str) -> LLMConfig | None:
        """Retrieve LLM configuration."""

    @abstractmethod
    def get_agent_config(self, agent_id: str) -> AgentConfig | None:
        """Retrieve agent configuration."""

    @abstractmethod
    def get_coordinator_config(self, coordinator_id: str) -> CoordinatorConfig | None:
        """Retrieve coordinator configuration."""

    @abstractmethod
    def get_session_info(self, session_id: str | None = None) -> SessionInfo | None:
        """Retrieve session information with all related configurations."""


class SQLiteChatLogs(ChatLogs):
    """A class for managing and storing chat message logs in SQLite.

    The class handles storage, retrieval, and versioning of chat messages,
    including support for message retries and status tracking. It also tracks
    LLM configurations, agents, coordinators, and sessions.
    """

    filename: str = param.String(default="lumen_chat_logs.db")

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.conn: sqlite3.Connection = sqlite3.connect(self.filename)
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        # Create schema_version table first to track schema changes
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                version TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Check if schema_version exists, if not insert current version
        self.cursor.execute("SELECT version FROM schema_info")
        version_row = self.cursor.fetchone()
        if not version_row:
            self.cursor.execute(
                "INSERT INTO schema_info (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )

        # Create messages table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                message_index INTEGER,
                message_user TEXT,
                message_json TEXT,
                memory_json TEXT,
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
                message_json TEXT NOT NULL,
                memory_json TEXT,
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
                tools JSON DEFAULT '[]',
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
                    "tools": json.dumps(normalize_value(getattr(agent, "tools", []))),
                    "prompts": json.dumps(normalize_value(agent.prompts)),
                    "template_overrides": json.dumps(
                        normalize_value(agent.template_overrides)
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
                "prompts": json.dumps(normalize_value(coordinator.prompts)),
                "template_overrides": json.dumps(
                    normalize_value(coordinator.template_overrides)
                ),
                "tools": json.dumps(normalize_value(getattr(coordinator, "tools", []))),
                "history": coordinator.history,
                "llm_id": llm_id,
            }
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO coordinators
                (coordinator_id, name, agents, prompts, template_overrides,
                 tools, history, llm_id)
                VALUES (
                    :coordinator_id, :name, :agents, :prompts, :template_overrides,
                    :tools, :history, :llm_id
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

    def upsert(
        self,
        message_id: str,
        session_id: str,
        message_index: int,
        message_user: str,
        message: ChatMessage | str,
        memory: dict | None = None,
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
        memory : dict, optional
            Memory state at the time of the message

        Notes
        -----
        If a message with the same message_id exists, it will be updated
        with the new content and timestamp
        """
        content_dict = self._serialize_message(message)
        content_json = json.dumps(content_dict)
        memory_json = json.dumps(normalize_value(dict(memory))) if memory else None

        try:
            self.cursor.execute(
                """
                INSERT INTO messages (
                    message_id, session_id, message_index,
                    message_user, message_json, memory_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    message_json = excluded.message_json,
                    message_index = excluded.message_index,
                    message_user = excluded.message_user,
                    memory_json = excluded.memory_json,
                    timestamp = CURRENT_TIMESTAMP
                """,
                (message_id, session_id, message_index, message_user, content_json, memory_json),
            )
            self.conn.commit()
        except Exception as e:
            log_debug(f"Failed to upsert message: {e!s}")

    def update_retry(
        self,
        message_id: str,
        message: ChatMessage | str,
        memory: dict | None = None,
    ) -> None:
        """Create a retry version of a message.

        Parameters
        ----------
        message_id : str
            The ID of the message to retry
        message : ChatMessage or str
            The new message content
        memory : dict, optional
            Memory state at the time of the retry

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
        content_dict = self._serialize_message(message)
        content_json = json.dumps(content_dict)

        # Serialize memory if provided
        memory_json = json.dumps(normalize_value(memory)) if memory else None

        # Check if message exists and get original content before starting transaction
        self.cursor.execute(
            """
            SELECT COUNT(*), message_json, memory_json, liked, disliked
            FROM messages
            WHERE message_id = ?
            """,
            (message_id,),
        )
        count, original_content, original_memory, original_liked, original_disliked = (
            self.cursor.fetchone()
        )

        if count == 0:
            raise ValueError(f"No original message found for message_id: {message_id}")

        # Compare as dictionaries to avoid false negatives due to JSON formatting
        try:
            original_dict = json.loads(original_content) if original_content else {}
            if content_dict == original_dict and (
                (memory is None and original_memory is None) or
                (memory is not None and original_memory is not None and
                 json.loads(original_memory) == normalize_value(memory))
            ):
                log_debug("Matching content and memory; no need to insert update")
                return
        except json.JSONDecodeError:
            # If original content is not valid JSON, it's definitely different
            pass

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
                        message_json,
                        memory_json,
                        liked,
                        disliked
                    )
                    VALUES (?, 0, ?, ?, ?, ?)
                    """,
                    (message_id, original_content, original_memory, original_liked, original_disliked),
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
                    message_json,
                    memory_json,
                    liked,
                    disliked
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, next_attempt, content_json, memory_json, prev_liked, prev_disliked),
            )

            # Update original message state and contents
            self.cursor.execute(
                """
                UPDATE messages
                SET state = 'edited',
                    message_json = ?,
                    memory_json = ?,
                    attempt_number = ?
                WHERE message_id = ?
                """,
                (content_json, memory_json, next_attempt, message_id),
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

    def view_messages(self, session_id: str | None = None) -> pd.DataFrame:
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
            message_id, message_index, message_user, message_json, memory_json,
            attempt_number, liked, disliked, state, timestamp
        """
        if session_id == "all":
            df = pd.read_sql("SELECT * FROM messages ORDER BY timestamp ASC", self.conn)
        else:
            if session_id is None:
                self.cursor.execute(
                    "SELECT session_id FROM messages ORDER BY timestamp LIMIT 1"
                )
                session_id = self.cursor.fetchone()[0]

            df = pd.read_sql(
                """
                SELECT message_id, message_index, message_user,
                        message_json, memory_json, attempt_number, liked, disliked, state, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY message_index ASC
                """,
                self.conn,
                params=(session_id,),
            )

        # Parse JSON message content
        if 'message_json' in df.columns:
            df['message_json'] = df['message_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) and x.strip().startswith('{') else x
            )

        # Parse JSON memory content
        if 'memory_json' in df.columns:
            df['memory_json'] = df['memory_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) and x.strip().startswith('{') else None
            )

        return df

    def view_message_history(self, message_id: str) -> pd.DataFrame:
        """Get the complete history of a message including all retries.

        Parameters
        ----------
        message_id : str
            The ID of the message to get history for

        Returns
        -------
        pandas.DataFrame
            DataFrame containing message history with columns:
            version, message_json, memory_json, liked, disliked, timestamp, revision_type
        """
        query = """
            SELECT
                attempt_number as version,
                message_json,
                memory_json,
                liked,
                disliked,
                timestamp,
                revision_type
            FROM retries
            WHERE message_id = ?
            ORDER BY timestamp
        """

        df_history = pd.read_sql(query, self.conn, params=(message_id,))

        # Parse JSON message content
        if 'message_json' in df_history.columns:
            df_history['message_json'] = df_history['message_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) and x.strip().startswith('{') else x
            )

        # Parse JSON memory content
        if 'memory_json' in df_history.columns:
            df_history['memory_json'] = df_history['memory_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) and x.strip().startswith('{') else None
            )

        return df_history

    def view_sessions(
        self, session_id: str | None = "all", username: str | None = None
    ) -> pd.DataFrame:
        """View sessions with their associated metrics from the database.

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
            - Basic session info: session_id, username, coordinator_id, created_at, last_active
            - User info (if available)
            - Message metrics: user_msgs, asst_msgs, etc.
            - Response time metrics
            - Interaction metrics
        """
        base_query = """
            WITH SessionMetrics AS (
                SELECT
                    m.session_id,
                    -- Drop column first_message and last_message from final output
                    MIN(m.timestamp) as _first_message,
                    MAX(m.timestamp) as _last_message,
                    -- Basic counts
                    CAST(SUM(CASE WHEN m.message_user = 'User' THEN 1 ELSE 0 END) AS INTEGER) as user_msgs,
                    CAST(SUM(CASE WHEN m.message_user != 'User' THEN 1 ELSE 0 END) AS INTEGER) as asst_msgs,
                    CAST(SUM(CASE WHEN m.liked = TRUE THEN 1 ELSE 0 END) AS INTEGER) as total_likes,
                    CAST(SUM(CASE WHEN m.disliked = TRUE THEN 1 ELSE 0 END) AS INTEGER) as total_dislikes,

                    -- Message complexity
                    CAST(ROUND(AVG(CASE WHEN m.message_user = 'User'
                        THEN LENGTH(m.message_json) - LENGTH(REPLACE(m.message_json, ' ', '')) + 1
                        END)) AS INTEGER) as avg_user_words,
                    CAST(ROUND(AVG(CASE WHEN m.message_user != 'User'
                        THEN LENGTH(m.message_json) - LENGTH(REPLACE(m.message_json, ' ', '')) + 1
                        END)) AS INTEGER) as avg_asst_words,
                    CAST(ROUND(AVG(LENGTH(m.message_json))) AS INTEGER) as avg_chars_per_msg,

                    -- Interaction patterns
                    CAST(COUNT(DISTINCT CASE WHEN m.state != 'initial' THEN m.message_id END) AS INTEGER) as edited_msgs,
                    CAST(COUNT(DISTINCT CASE WHEN m.state = 'retried' THEN m.message_id END) AS INTEGER) as retried_msgs,
                    CAST(COUNT(DISTINCT CASE WHEN m.state = 'undone' THEN m.message_id END) AS INTEGER) as undone_msgs,

                    -- Response times
                    CAST(ROUND(AVG(CASE
                        WHEN m.message_user != 'User' AND m2.message_user = 'User'
                        THEN (julianday(m.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60
                    END)) AS INTEGER) as avg_asst_resp_s,
                    CAST(MIN(CASE
                        WHEN m.message_user != 'User' AND m2.message_user = 'User'
                        THEN (julianday(m.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60
                    END) AS INTEGER) as min_asst_resp_s,
                    CAST(MAX(CASE
                        WHEN m.message_user != 'User' AND m2.message_user = 'User'
                        THEN (julianday(m.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60
                    END) AS INTEGER) as max_asst_resp_s,
                    CAST(ROUND(AVG(CASE
                        WHEN m.message_user = 'User' AND m2.message_user != 'User'
                        THEN (julianday(m.timestamp) - julianday(m2.timestamp)) * 24 * 60 * 60
                    END)) AS INTEGER) as avg_user_resp_s,

                    -- Lumen-specific metrics
                    CAST(ROUND(AVG(CASE
                        WHEN m.message_user = 'Lumen' AND m2.message_user = 'Assistant'
                        THEN (julianday(m2.timestamp) - julianday(m.timestamp)) * 24 * 60 * 60
                    END)) AS INTEGER) as avg_lumen_resp_s,
                    CAST(MIN(CASE
                        WHEN m.message_user = 'Lumen' AND m2.message_user = 'Assistant'
                        THEN (julianday(m2.timestamp) - julianday(m.timestamp)) * 24 * 60 * 60
                    END) AS INTEGER) as min_lumen_resp_s,
                    CAST(MAX(CASE
                        WHEN m.message_user = 'Lumen' AND m2.message_user = 'Assistant'
                        THEN (julianday(m2.timestamp) - julianday(m.timestamp)) * 24 * 60 * 60
                    END) AS INTEGER) as max_lumen_resp_s
                FROM messages m
                LEFT JOIN messages m2 ON m2.message_index = m.message_index - 1
                    AND m2.session_id = m.session_id
                GROUP BY m.session_id
            )
            SELECT
                s.session_id,
                s.username,
                s.coordinator_id,
                s.created_at,
                s.last_active,
                s.user_info,
                CAST(ROUND((julianday(sm._last_message) - julianday(sm._first_message)) * 24 * 60 * 60) AS INTEGER) as duration_s,
                sm.user_msgs,
                sm.asst_msgs,
                sm.total_likes,
                sm.total_dislikes,
                sm.avg_user_words,
                sm.avg_asst_words,
                sm.avg_chars_per_msg,
                sm.edited_msgs,
                sm.retried_msgs,
                sm.undone_msgs,
                sm.avg_asst_resp_s,
                sm.min_asst_resp_s,
                sm.max_asst_resp_s,
                sm.avg_user_resp_s,
                sm.avg_lumen_resp_s,
                sm.min_lumen_resp_s,
                sm.max_lumen_resp_s
            FROM sessions s
            LEFT JOIN SessionMetrics sm ON s.session_id = sm.session_id
        """

        # Handle different session_id scenarios
        if session_id == "all":
            if username is not None:
                query = base_query + " WHERE s.username = ? ORDER BY s.created_at DESC"
                params = [username]
            else:
                query = base_query + " ORDER BY s.created_at DESC"
                params = []
        else:
            if session_id is None:
                # Get the most recent session
                self.cursor.execute(
                    "SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT 1"
                )
                result = self.cursor.fetchone()
                if result is None:
                    # Return empty DataFrame with correct columns
                    return pd.DataFrame(
                        columns=[
                            "session_id",
                            "username",
                            "coordinator_id",
                            "created_at",
                            "last_active",
                            "user_info",
                            "duration_s",
                            "user_msgs",
                            "asst_msgs",
                            "total_likes",
                            "total_dislikes",
                            "avg_user_words",
                            "avg_asst_words",
                            "avg_chars_per_msg",
                            "edited_msgs",
                            "retried_msgs",
                            "undone_msgs",
                            "avg_asst_resp_s",
                            "min_asst_resp_s",
                            "max_asst_resp_s",
                            "avg_user_resp_s",
                            "avg_lumen_resp_s",
                            "min_lumen_resp_s",
                            "max_lumen_resp_s",
                        ]
                    )
                session_id = result[0]

            query = base_query + " WHERE s.session_id = ?"
            params = [session_id]

        # Execute query and get results
        df = pd.read_sql(query, self.conn, params=params)

        # Parse JSON user_info if present
        if "user_info" in df.columns:
            df["user_info"] = df["user_info"].apply(
                lambda x: json.loads(x) if pd.notna(x) else None
            )

        # Set session_id as index
        return df.set_index("session_id")

    def view_coordinator_agents(
        self, coordinator_id: str | None = None
    ) -> pd.DataFrame:
        """View detailed information about all agents in a coordinator.

        Parameters
        ----------
        coordinator_id : str or None, optional
            The coordinator ID to retrieve information for.
            If None, returns information for the most recent coordinator.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing detailed agent information with prompts as dictionaries.
        """
        if coordinator_id is None:
            self.cursor.execute(
                "SELECT coordinator_id FROM coordinators ORDER BY created_at DESC LIMIT 1"
            )
            result = self.cursor.fetchone()
            if result is None:
                return pd.DataFrame()
            coordinator_id = result[0]

        coordinator = self.get_coordinator_config(coordinator_id)
        if not coordinator:
            return pd.DataFrame()

        # Create agent rows with prompts as dictionaries
        agents_data = []
        for agent in coordinator.agents:
            # Extract response_model if it exists in prompts
            response_model = agent.prompts.get("response_model", None)
            prompts = {k: v for k, v in agent.prompts.items() if k != "response_model"}

            agent_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "purpose": agent.purpose.strip() if agent.purpose else None,
                "requires": agent.requires,
                "provides": agent.provides,
                "tools": agent.tools,
                "llm_id": agent.llm.llm_id if agent.llm else None,
                "llm_mode": agent.llm.mode if agent.llm else None,
                "llm_model_args": agent.llm.model_kwargs if agent.llm else None,
                "llm_temperature": agent.llm.temperature if agent.llm else None,
                "response_model": response_model,
                "prompts": prompts,
                "template_overrides": agent.template_overrides,
            }
            agents_data.append(agent_data)

        df = pd.DataFrame(agents_data)

        # Define column order
        columns = [
            "agent_id",
            "name",
            "purpose",
            "requires",
            "provides",
            "tools",
            "llm_id",
            "llm_mode",
            "llm_model_args",
            "llm_temperature",
            "response_model",
            "prompts",
            "template_overrides",
        ]

        # Ensure all columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = None

        return df[columns]

    def get_llm_config(self, llm_id: str) -> LLMConfig | None:
        """Retrieve LLM configuration."""
        self.cursor.execute(
            """
            SELECT llm_id, mode, model_kwargs, temperature, created_at
            FROM llm_configs
            WHERE llm_id = ?
            """,
            (llm_id,),
        )
        row = self.cursor.fetchone()

        if not row:
            return None

        return LLMConfig(
            llm_id=row[0],
            mode=row[1],
            model_kwargs=json.loads(row[2]) if row[2] else {},
            temperature=row[3],
            created_at=datetime.fromisoformat(row[4]),
        )

    def get_agent_config(self, agent_id: str) -> AgentConfig | None:
        """Retrieve agent configuration."""
        self.cursor.execute(
            """
            SELECT agent_id, name, purpose, requires, provides, tools,
                prompts, template_overrides, llm_id, created_at
            FROM agents
            WHERE agent_id = ?
            """,
            (agent_id,),
        )
        row = self.cursor.fetchone()

        if not row:
            return None

        llm_config = self.get_llm_config(row[8]) if row[8] else None

        return AgentConfig(
            agent_id=row[0],
            name=row[1],
            purpose=row[2],
            requires=json.loads(row[3]),
            provides=json.loads(row[4]),
            tools=json.loads(row[5]),
            prompts=json.loads(row[6]),
            template_overrides=json.loads(row[7]),
            llm=llm_config,
            created_at=datetime.fromisoformat(row[9]),
        )

    def get_coordinator_config(self, coordinator_id: str) -> CoordinatorConfig | None:
        """Retrieve coordinator configuration."""
        self.cursor.execute(
            """
            SELECT coordinator_id, name, agents, prompts,
                template_overrides, tools, history, llm_id, created_at
            FROM coordinators
            WHERE coordinator_id = ?
            """,
            (coordinator_id,),
        )
        row = self.cursor.fetchone()

        if not row:
            return None

        # Get LLM config if present
        llm_config = self.get_llm_config(row[7]) if row[7] else None

        # Get agent configs
        agent_ids = json.loads(row[2])
        agents = [
            self.get_agent_config(agent_id)
            for agent_id in agent_ids
            if (_agent := self.get_agent_config(agent_id)) is not None
        ]

        return CoordinatorConfig(
            coordinator_id=row[0],
            name=row[1],
            agents=agents,
            prompts=json.loads(row[3]),
            template_overrides=json.loads(row[4]),
            tools=json.loads(row[5]),
            history=row[6],
            llm=llm_config,
            created_at=datetime.fromisoformat(row[8]),
        )

    def get_session_info(self, session_id: str | None = None) -> SessionInfo | None:
        """Retrieve session information with all related configurations.

        Parameters
        ----------
        session_id : str, optional
            The session ID to retrieve. If None, returns the most recent session.

        Returns
        -------
        Optional[SessionInfo]
            Complete session information including coordinator and agent configs.
        """
        # Use existing get_sessions() method to get session info and metrics
        df = self.view_sessions(session_id=session_id)
        if df.empty:
            return None

        # Get the single row of session data
        session_data = df.iloc[0]

        # Get coordinator config
        coordinator = self.get_coordinator_config(session_data.coordinator_id)
        if not coordinator:
            return None

        # Create metrics DataFrame with just the metrics columns
        metrics_columns = [
            "duration_s",
            "user_msgs",
            "asst_msgs",
            "total_likes",
            "total_dislikes",
            "avg_user_words",
            "avg_asst_words",
            "avg_chars_per_msg",
            "edited_msgs",
            "retried_msgs",
            "undone_msgs",
            "avg_asst_resp_s",
            "min_asst_resp_s",
            "max_asst_resp_s",
            "avg_user_resp_s",
            "avg_lumen_resp_s",
            "min_lumen_resp_s",
            "max_lumen_resp_s",
        ]
        metrics = df[metrics_columns].iloc[0].to_dict()

        return SessionInfo(
            session_id=session_data.name,  # Index is session_id
            username=session_data.username,
            user_info=session_data.user_info,
            coordinator=coordinator,
            created_at=datetime.fromisoformat(session_data.created_at),
            last_active=datetime.fromisoformat(session_data.last_active),
            metrics=metrics,
        )

    def get_schema_version(self) -> str:
        """Get the current schema version from the database."""
        self.cursor.execute("SELECT version FROM schema_info LIMIT 1")
        result = self.cursor.fetchone()
        return result[0] if result else SCHEMA_VERSION


# Schema migration function for future use
def migrate_schema(from_version: str, to_version: str, db_connection: Any) -> bool:
    """Migrate database schema from one version to another.

    Parameters
    ----------
    from_version : str
        Current schema version
    to_version : str
        Target schema version
    db_connection : Any
        Database connection to apply migrations to

    Returns
    -------
    bool
        True if migration was successful, False otherwise
    """
    if from_version == to_version:
        return True

    # Example migrations could be implemented here
    # if from_version == "1.0.0" and to_version == "1.1.0":
    #     # Apply migration steps for 1.0.0 -> 1.1.0
    #     return True

    return False
