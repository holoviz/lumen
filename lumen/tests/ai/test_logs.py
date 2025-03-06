import json
import os

from unittest.mock import MagicMock

import pandas as pd
import pytest

from panel import Card
from panel.chat import ChatMessage, ChatStep
from panel.widgets import Tabulator

try:
    from lumen.ai.logs import (
        SCHEMA_VERSION, AgentConfig, CoordinatorConfig, LLMConfig, SessionInfo,
        SQLiteChatLogs,
    )
except ImportError:
    pytest.skip("Skipping tests that require lumen.ai", allow_module_level=True)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database file path."""
    return str(tmp_path / "test_chat_logs.db")


@pytest.fixture
def chat_logs(temp_db_path):
    """Create a SQLiteChatLogs instance with a temporary database."""
    logs = SQLiteChatLogs(filename=temp_db_path)
    yield logs
    # Close connection to prevent "database is locked" issues
    logs.conn.close()
    # Clean up temp file
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except PermissionError:
            pass


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with agents and LLMs."""
    coordinator = MagicMock()
    coordinator.hash = "test_coordinator_id"
    coordinator.interface._session_id = "test_session_id"
    coordinator.history = 3
    coordinator.prompts = {"main": {"template": "test template"}}
    coordinator.template_overrides = {}
    coordinator.tools = ["tool1", "tool2"]

    # Mock LLM
    llm = MagicMock()
    llm.hash = "test_llm_id"
    llm.mode = "Mode.TOOLS"
    llm.model_kwargs = {"model": "gpt-4o-mini"}
    llm.temperature = 0.25
    coordinator.llm = llm

    # Mock Agents
    agent1 = MagicMock()
    agent1.hash = "test_agent_id_1"
    agent1.requires = ["source"]
    agent1.provides = ["data"]
    agent1.tools = []
    agent1.prompts = {"main": {"template": "agent template"}}
    agent1.template_overrides = {}
    agent1.purpose = "This is a test agent"

    agent2 = MagicMock()
    agent2.hash = "test_agent_id_2"
    agent2.requires = []
    agent2.provides = ["sql"]
    agent2.tools = []
    agent2.prompts = {"main": {"template": "agent 2 template"}}
    agent2.template_overrides = {}
    agent2.purpose = "This is another test agent"

    coordinator.agents = [agent1, agent2]

    return coordinator


class TestSQLiteChatLogs:
    """Tests for the ChatLogs class."""

    def test_initialization(self, chat_logs, temp_db_path):
        """Test that the database is properly initialized."""
        # Check that the database file is created
        assert os.path.exists(temp_db_path)

        # Check that tables are created
        tables = chat_logs.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [table[0] for table in tables]

        expected_tables = [
            "schema_info", "messages", "retries", "llm_configs", "agents",
            "coordinators", "sessions"
        ]
        for table in expected_tables:
            assert table in table_names

        # Check schema version
        version = chat_logs.get_schema_version()
        assert version == SCHEMA_VERSION

    def test_register_coordinator(self, chat_logs, mock_coordinator):
        """Test registering a coordinator and its components."""
        session_id = chat_logs.register_coordinator(mock_coordinator)

        # Check that session was created
        assert session_id == "test_session_id"

        # Check that LLM was registered
        llm_configs = chat_logs.cursor.execute(
            "SELECT * FROM llm_configs WHERE llm_id = ?",
            ("test_llm_id",)
        ).fetchall()
        assert len(llm_configs) == 1

        # Check that agents were registered
        agents = chat_logs.cursor.execute(
            "SELECT * FROM agents WHERE agent_id IN (?, ?)",
            ("test_agent_id_1", "test_agent_id_2")
        ).fetchall()
        assert len(agents) == 2

        # Check that coordinator was registered
        coordinators = chat_logs.cursor.execute(
            "SELECT * FROM coordinators WHERE coordinator_id = ?",
            ("test_coordinator_id",)
        ).fetchall()
        assert len(coordinators) == 1

        # Check that session links to coordinator
        sessions = chat_logs.cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            ("test_session_id",)
        ).fetchall()
        assert len(sessions) == 1
        assert sessions[0][3] == "test_coordinator_id"  # coordinator_id column

    def test_upsert_message(self, chat_logs):
        """Test inserting and updating messages."""
        # Insert a new message
        chat_logs.upsert(
            message_id="test_msg_1",
            session_id="test_session",
            message_index=0,
            message_user="User",
            message="Hello, world!"
        )

        # Check that message was inserted
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_1",)
        ).fetchall()
        assert len(messages) == 1
        assert messages[0][2] == 0  # message_index
        assert messages[0][3] == "User"  # message_user

        # Verify the message is stored as JSON
        message_json = json.loads(messages[0][4])  # message_json as JSON
        assert message_json["type"] == "text"
        assert message_json["content"] == "Hello, world!"

        # Update the message
        chat_logs.upsert(
            message_id="test_msg_1",
            session_id="test_session",
            message_index=0,
            message_user="User",
            message="Updated message"
        )

        # Check that message was updated
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_1",)
        ).fetchall()
        assert len(messages) == 1

        # Verify the message content was updated
        message_json = json.loads(messages[0][4])  # message_json as JSON
        assert message_json["type"] == "text"
        assert message_json["content"] == "Updated message"

    def test_update_retry(self, chat_logs):
        """Test creating retry versions of messages."""
        # Insert a message first
        chat_logs.upsert(
            message_id="test_msg_retry",
            session_id="test_session",
            message_index=0,
            message_user="Assistant",
            message="Initial response"
        )

        # Create a retry
        chat_logs.update_retry(
            message_id="test_msg_retry",
            message="Retry response"
        )

        # Check that original message was updated
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_retry",)
        ).fetchall()
        assert len(messages) == 1

        # Verify the message content was updated
        message_json = json.loads(messages[0][4])  # message_json as JSON
        assert message_json["type"] == "text"
        assert message_json["content"] == "Retry response"

        assert messages[0][6] == 1  # attempt_number
        assert messages[0][9] == "edited"  # state

        # Check that retry was recorded
        retries = chat_logs.cursor.execute(
            "SELECT * FROM retries WHERE message_id = ?",
            ("test_msg_retry",)
        ).fetchall()
        assert len(retries) == 2  # Original + retry

        # Add another retry
        chat_logs.update_retry(
            message_id="test_msg_retry",
            message="Second retry"
        )

        # Check that message was updated again
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_retry",)
        ).fetchall()

        # Verify the message content was updated
        message_json = json.loads(messages[0][4])  # message_json as JSON
        assert message_json["type"] == "text"
        assert message_json["content"] == "Second retry"
        assert messages[0][6] == 2  # attempt_number

        # Check that all retries were recorded
        retries = chat_logs.cursor.execute(
            "SELECT * FROM retries WHERE message_id = ? ORDER BY attempt_number",
            ("test_msg_retry",)
        ).fetchall()
        assert len(retries) == 3  # Original + 2 retries
        assert retries[0][2] == 0  # attempt_number (original)
        assert retries[1][2] == 1  # attempt_number (first retry)
        assert retries[2][2] == 2  # attempt_number (second retry)

    def test_update_status(self, chat_logs):
        """Test updating message status (liked, disliked, state)."""
        # Insert a message first
        chat_logs.upsert(
            message_id="test_msg_status",
            session_id="test_session",
            message_index=0,
            message_user="Assistant",
            message="Status test message"
        )

        # Update liked status
        chat_logs.update_status(
            message_id="test_msg_status",
            liked=True
        )

        # Check that status was updated
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_status",)
        ).fetchall()
        assert messages[0][7] == True  # liked

        # Update disliked status
        chat_logs.update_status(
            message_id="test_msg_status",
            disliked=True
        )

        # Check that status was updated
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_status",)
        ).fetchall()
        assert messages[0][8] == True  # disliked

        # Update state
        chat_logs.update_status(
            message_id="test_msg_status",
            state="retried"
        )

        # Check that state was updated
        messages = chat_logs.cursor.execute(
            "SELECT * FROM messages WHERE message_id = ?",
            ("test_msg_status",)
        ).fetchall()
        assert messages[0][9] == "retried"  # state

    def test_view_messages(self, chat_logs):
        """Test retrieving messages for a session."""
        # Insert test messages
        chat_logs.upsert(
            message_id="test_view_1",
            session_id="test_view_session",
            message_index=0,
            message_user="User",
            message="User message 1"
        )

        chat_logs.upsert(
            message_id="test_view_2",
            session_id="test_view_session",
            message_index=1,
            message_user="Assistant",
            message="Assistant reply 1"
        )

        chat_logs.upsert(
            message_id="test_view_3",
            session_id="test_view_session",
            message_index=2,
            message_user="User",
            message="User message 2"
        )

        # Test viewing messages for a session
        df = chat_logs.view_messages("test_view_session")

        # Verify dataframe content
        assert len(df) == 3
        assert list(df["message_user"]) == ["User", "Assistant", "User"]

        # Check message content in the structured format
        assert df["message_json"][0]["type"] == "text"
        assert df["message_json"][0]["content"] == "User message 1"
        assert df["message_json"][1]["type"] == "text"
        assert df["message_json"][1]["content"] == "Assistant reply 1"
        assert df["message_json"][2]["type"] == "text"
        assert df["message_json"][2]["content"] == "User message 2"

    def test_view_message_history(self, chat_logs):
        """Test retrieving message history including retries."""
        # Insert a message
        chat_logs.upsert(
            message_id="test_history",
            session_id="test_session",
            message_index=0,
            message_user="Assistant",
            message="Original response"
        )

        # Add retries
        chat_logs.update_retry(
            message_id="test_history",
            message="First retry"
        )

        chat_logs.update_retry(
            message_id="test_history",
            message="Second retry"
        )

        # Add like to latest version
        chat_logs.update_status(
            message_id="test_history",
            liked=True
        )

        # Get message history
        df_history = chat_logs.view_message_history("test_history")

        # Verify dataframe content
        assert len(df_history) == 3
        assert list(df_history["version"]) == [0, 1, 2]

        # Check message content in each version
        assert df_history["message_json"][0]["type"] == "text"
        assert df_history["message_json"][0]["content"] == "Original response"
        assert df_history["message_json"][1]["type"] == "text"
        assert df_history["message_json"][1]["content"] == "First retry"
        assert df_history["message_json"][2]["type"] == "text"
        assert df_history["message_json"][2]["content"] == "Second retry"

        # Latest version should be liked
        assert df_history.iloc[2]["liked"] == True

    def test_get_session_info(self, chat_logs, mock_coordinator):
        """Test retrieving session information."""
        # Register coordinator and add some messages
        chat_logs.register_coordinator(mock_coordinator)

        chat_logs.upsert(
            message_id="test_session_1",
            session_id="test_session_id",
            message_index=0,
            message_user="User",
            message="User message"
        )

        chat_logs.upsert(
            message_id="test_session_2",
            session_id="test_session_id",
            message_index=1,
            message_user="Assistant",
            message="Assistant reply"
        )

        # Get session info
        session_info = chat_logs.get_session_info("test_session_id")

        # Verify session info
        assert isinstance(session_info, SessionInfo)
        assert session_info.session_id == "test_session_id"
        assert session_info.username == "anonymous"
        assert session_info.coordinator is not None
        assert "user_msgs" in session_info.metrics
        assert "asst_msgs" in session_info.metrics

    def test_view_sessions(self, chat_logs, mock_coordinator):
        """Test retrieving session metrics."""
        # Register coordinator and add some messages
        chat_logs.register_coordinator(mock_coordinator, username="test_user")

        # Add messages with timestamps
        chat_logs.upsert(
            message_id="test_sessions_1",
            session_id="test_session_id",
            message_index=0,
            message_user="User",
            message="User message 1"
        )

        chat_logs.upsert(
            message_id="test_sessions_2",
            session_id="test_session_id",
            message_index=1,
            message_user="Assistant",
            message="Assistant reply 1"
        )

        # Get session metrics
        df_sessions = chat_logs.view_sessions("test_session_id")

        # Verify basic metrics
        assert not df_sessions.empty
        assert df_sessions.index[0] == "test_session_id"
        assert df_sessions.iloc[0]["username"] == "test_user"
        assert df_sessions.iloc[0]["user_msgs"] == 1
        assert df_sessions.iloc[0]["asst_msgs"] == 1

    def test_view_coordinator_agents(self, chat_logs, mock_coordinator):
        """Test retrieving coordinator agent information."""
        # Register coordinator to set up test data
        chat_logs.register_coordinator(mock_coordinator)

        # Get coordinator agents
        df_agents = chat_logs.view_coordinator_agents("test_coordinator_id")

        # Verify dataframe content
        assert len(df_agents) == 2
        assert list(df_agents["agent_id"]) == ["test_agent_id_1", "test_agent_id_2"]
        assert list(df_agents["llm_id"]) == ["test_llm_id", "test_llm_id"]

        # Check that requires/provides are parsed correctly
        assert df_agents.iloc[0]["requires"] == ["source"]
        assert df_agents.iloc[0]["provides"] == ["data"]
        assert df_agents.iloc[1]["provides"] == ["sql"]

    def test_serialize_message_string(self, chat_logs):
        """Test serializing string messages."""
        result = chat_logs._serialize_message("Simple text message")
        assert result["type"] == "text"
        assert result["content"] == "Simple text message"

    def test_serialize_message_chat_message(self, chat_logs):
        """Test serializing ChatMessage objects."""
        # Create an actual ChatMessage with text
        msg = ChatMessage("Test message content")

        # Test serialization
        result = chat_logs._serialize_message(msg)
        assert result["type"] == "message"
        assert "Test message content" in result["content"]

    def test_serialize_message_card(self, chat_logs):
        """Test serializing Card objects within ChatMessage."""
        # Create actual Card with steps
        step1 = ChatStep(title="Step 1")
        step1.append("Step 1 content")

        step2 = ChatStep(title="Step 2")
        step2.append("Step 2 content")

        card = Card(title="Test Card")
        card.append(step1)
        card.append(step2)

        # Create actual ChatMessage with Card
        msg = ChatMessage(card)

        # Test serialization
        result = chat_logs._serialize_message(msg)
        assert result["type"] == "card"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["title"] == "Step 1"
        assert "Step 1 content" in result["steps"][0]["content"]
        assert result["steps"][1]["title"] == "Step 2"
        assert "Step 2 content" in result["steps"][1]["content"]

    def test_serialize_message_tabulator(self, chat_logs):
        """Test serializing Tabulator objects within ChatMessage."""
        # Create actual Tabulator with DataFrame
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        tabulator = Tabulator(df)

        # Create actual ChatMessage with Tabulator
        msg = ChatMessage(tabulator)

        # Test serialization
        result = chat_logs._serialize_message(msg)
        assert result["type"] == "table"
        assert "data" in result
        assert len(result["data"]) == 3  # 3 rows

    def test_serialize_message_to_string(self, chat_logs):
        """Test converting serialized message back to string."""
        # Test text message
        msg = "Simple text message"
        message_dict = chat_logs._serialize_message(msg)
        string_result = chat_logs._serialize_message_to_string(message_dict)
        assert string_result == "Simple text message"

        # Test card message
        step1 = ChatStep(title="Step 1")
        step1.append("Step 1 content")
        card = Card(title="Test Card")
        card.append(step1)
        msg = ChatMessage(card)
        message_dict = chat_logs._serialize_message(msg)
        string_result = chat_logs._serialize_message_to_string(message_dict)
        assert "Step 1" in string_result
        assert "Step 1 content" in string_result

    def test_view_message_all_sessions(self, chat_logs):
        """Test retrieving messages from all sessions."""
        # Add messages to different sessions
        chat_logs.upsert(
            message_id="test_all_1",
            session_id="session1",
            message_index=0,
            message_user="User",
            message="Session 1 message"
        )

        chat_logs.upsert(
            message_id="test_all_2",
            session_id="session2",
            message_index=0,
            message_user="User",
            message="Session 2 message"
        )

        # Get all messages
        df = chat_logs.view_messages("all")

        # Verify dataframe includes messages from both sessions
        assert len(df) == 2
        session_ids = list(set(df["session_id"]))
        assert "session1" in session_ids
        assert "session2" in session_ids

    def test_empty_session_info(self, chat_logs):
        """Test behavior when requesting non-existent session info."""
        # Try to get info for non-existent session
        session_info = chat_logs.get_session_info("non_existent_session")

        # Should return None
        assert session_info is None

    def test_custom_user_info(self, chat_logs, mock_coordinator):
        """Test storing and retrieving custom user info."""
        # Custom user info
        user_info = {
            "client_version": "1.2.3",
            "custom_metrics": {
                "feature_usage": {"code_generation": 5}
            }
        }

        # Register coordinator with user info
        chat_logs.register_coordinator(
            mock_coordinator,
            username="custom_user",
            user_info=user_info
        )

        # Get session info
        df_sessions = chat_logs.view_sessions("test_session_id")

        # Check user info was stored correctly
        assert df_sessions.iloc[0]["username"] == "custom_user"
        assert df_sessions.iloc[0]["user_info"] == user_info

    def test_error_handling_non_existent_message(self, chat_logs):
        """Test error handling for non-existent messages."""
        # Try to update a non-existent message
        with pytest.raises(ValueError):
            chat_logs.update_retry(
                message_id="non_existent_message",
                message="This should fail"
            )

    def test_no_retry_for_identical_content(self, chat_logs):
        """Test that retry isn't created for identical content."""
        # Insert a message
        chat_logs.upsert(
            message_id="test_identical",
            session_id="test_session",
            message_index=0,
            message_user="Assistant",
            message="Same content"
        )

        # Try to update with identical content
        chat_logs.update_retry(
            message_id="test_identical",
            message="Same content"
        )

        # Check that no retry was recorded
        retries = chat_logs.cursor.execute(
            "SELECT * FROM retries WHERE message_id = ?",
            ("test_identical",)
        ).fetchall()
        assert len(retries) == 0

    def test_multiple_status_updates(self, chat_logs):
        """Test updating multiple status fields at once."""
        # Insert a message
        chat_logs.upsert(
            message_id="test_multi_status",
            session_id="test_session",
            message_index=0,
            message_user="Assistant",
            message="Multi-status test"
        )

        # Update multiple fields
        chat_logs.update_status(
            message_id="test_multi_status",
            liked=True,
            disliked=False,
            state="retried"
        )

        # Check all fields were updated
        messages = chat_logs.cursor.execute(
            "SELECT liked, disliked, state FROM messages WHERE message_id = ?",
            ("test_multi_status",)
        ).fetchall()
        assert messages[0][0] == True  # liked
        assert messages[0][1] == False  # disliked
        assert messages[0][2] == "retried"  # state
