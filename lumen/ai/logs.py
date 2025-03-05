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

from lumen.ai.utils import log_debug, serialize_value

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
class ContextProviderConfig:
    """Configuration for a context provider."""

    provides: list[str]
    purpose: str | None
    requires: list[str]

    def pprint(self):
        pprint(self.__dict__, width=PPRINT_WIDTH)


@dataclass
class ActorConfig:
    """Configuration for an actor in the system."""

    id: str
    name: str
    prompts: dict[str, Any]
    template_overrides: dict[str, Any]
    llm: LLMConfig | None
    params: dict[str, Any]
    created_at: datetime

    def pprint(self):
        pprint(self.__dict__, width=PPRINT_WIDTH)


@dataclass
class ToolConfig(ActorConfig, ContextProviderConfig):
    """Configuration for a tool in the system."""


@dataclass
class AgentConfig(ActorConfig, ContextProviderConfig):
    """Configuration for an agent in the system."""

    tools: list[ToolConfig]


@dataclass
class CoordinatorConfig(ActorConfig):
    """Configuration for a coordinator managing multiple agents."""

    id: str
    agents: list[AgentConfig]
    tools: list[ToolConfig]
    history: int


@dataclass
class SessionInfo:
    """Information about a chat session."""

    id: str
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
                step_dict["status"] = step.status
                steps.append(step_dict)

            return {
                "type": "card",
                "steps": steps,
                "user": message.user,
            }

        elif isinstance(message.object, LumenOutput):
            output = message.object
            output_data = {
                "type": "lumen_output",
                "output_type": output.__class__.__name__,
                "title": output.title,
                "spec": output.spec,
                "language": output.language,
                "active": output.active,
                "user": message.user,
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
                "user": message.user,
            }

        return {
            "type": "message",
            "content": message.serialize(),
            "user": message.user,
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
    def get_tool_config(self, tool_id: str) -> ToolConfig | None:
        """Retrieve tool configuration."""

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

        # Create tools table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                tool_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                purpose TEXT,
                requires JSON DEFAULT '[]',
                provides JSON DEFAULT '[]',
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
                agent_ids JSON NOT NULL,
                prompts JSON NOT NULL,
                template_overrides JSON DEFAULT '{}',
                tool_ids JSON DEFAULT '[]',
                history INTEGER DEFAULT 3,
                llm_id TEXT REFERENCES llm_configs(llm_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, prompts, template_overrides, llm_id, agent_ids)
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
            "CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_llm_id ON agents(llm_id)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_llm_id ON tools(llm_id)"
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
        3. Register all tools
        4. Register the coordinator
        5. Create a new session
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
                    "tools": json.dumps(serialize_value(getattr(agent, "tools", []))),
                    "prompts": json.dumps(serialize_value(agent.prompts)),
                    "template_overrides": json.dumps(
                        serialize_value(agent.template_overrides)
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

            # Register tools
            tool_ids = []
            tools = getattr(coordinator, "tools", [])
            for tool in tools:
                tool_data = {
                    "tool_id": tool.hash,
                    "name": type(tool).__name__,
                    "purpose": getattr(tool, "purpose", None),
                    "requires": json.dumps(list(tool.requires)),
                    "provides": json.dumps(list(tool.provides)),
                    "prompts": json.dumps(serialize_value(tool.prompts)),
                    "template_overrides": json.dumps(
                        serialize_value(tool.template_overrides)
                    ),
                    "llm_id": llm_id,
                }
                self.cursor.execute(
                    """
                    INSERT OR IGNORE INTO tools
                    (tool_id, name, purpose, requires, provides,
                     prompts, template_overrides, llm_id)
                    VALUES (
                        :tool_id, :name, :purpose, :requires, :provides,
                        :prompts, :template_overrides, :llm_id
                    )
                    """,
                    tool_data,
                )
                tool_ids.append(tool.hash)

            # Register coordinator
            coordinator_data = {
                "coordinator_id": coordinator.hash,
                "name": type(coordinator).__name__,
                "agent_ids": json.dumps(agent_ids),
                "prompts": json.dumps(serialize_value(coordinator.prompts)),
                "template_overrides": json.dumps(
                    serialize_value(coordinator.template_overrides)
                ),
                "tool_ids": json.dumps(tool_ids),
                "history": coordinator.history,
                "llm_id": llm_id,
            }
            self.cursor.execute(
                """
                INSERT OR IGNORE INTO coordinators
                (coordinator_id, name, agent_ids, prompts, template_overrides,
                 tool_ids, history, llm_id)
                VALUES (
                    :coordinator_id, :name, :agent_ids, :prompts, :template_overrides,
                    :tool_ids, :history, :llm_id
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

    def get_tool_config(self, tool_id: str) -> ToolConfig | None:
        """Retrieve tool configuration."""
        self.cursor.execute(
            """
            SELECT tool_id, name, purpose, requires, provides,
                prompts, template_overrides, llm_id, created_at
            FROM tools
            WHERE tool_id = ?
            """,
            (tool_id,),
        )
        row = self.cursor.fetchone()

        if not row:
            return None

        llm_config = self.get_llm_config(row[7]) if row[7] else None

        return ToolConfig(
            tool_id=row[0],
            name=row[1],
            purpose=row[2],
            requires=json.loads(row[3]),
            provides=json.loads(row[4]),
            prompts=json.loads(row[5]),
            template_overrides=json.loads(row[6]),
            llm=llm_config,
            created_at=datetime.fromisoformat(row[8]),
        )

    def get_coordinator_config(self, coordinator_id: str) -> CoordinatorConfig | None:
        """Retrieve coordinator configuration."""
        self.cursor.execute(
            """
            SELECT coordinator_id, name, agent_ids, prompts,
                template_overrides, tool_ids, history, llm_id, created_at
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

        # Get tool configs
        tool_ids = json.loads(row[5])
        tools = [
            self.get_tool_config(tool_id)
            for tool_id in tool_ids
            if (_tool := self.get_tool_config(tool_id)) is not None
        ]

        return CoordinatorConfig(
            coordinator_id=row[0],
            name=row[1],
            agent_ids=agents,
            prompts=json.loads(row[3]),
            template_overrides=json.loads(row[4]),
            tool_ids=tools,
            history=row[6],
            llm=llm_config,
            created_at=datetime.fromisoformat(row[8]),
        )

    def get_session_info(self, session_id: str | None = None) -> SessionInfo | None:
        """Retrieve session information with all related configurations."""
        # Get session info and metrics using existing view_sessions method
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
