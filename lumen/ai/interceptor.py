import datetime
import sqlite3

from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Dict, List,
)

import param

if TYPE_CHECKING:
    import openai


class OpenAIInterceptor(param.Parameterized):

    db_path = param.String(
        default="messages.db", doc="Path to the SQLite database file"
    )

    def __init__(self, **params):
        super().__init__(**params)
        needs_init = not Path(self.db_path).exists()
        self.conn: sqlite3.Connection = self._create_connection()
        if needs_init:
            self.init_db()
        self._client = self._original_create = None
        self.session_id: str = self._generate_session_id()

    def _create_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        first_message_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"conv_{int(first_message_time)}"

    def init_db(self) -> None:
        """
        Initialize the database by creating necessary tables if they don't exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS message_groups (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                group_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES message_groups(id)
            )
            """
        )
        self.conn.commit()

    def store_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Store messages in the database for the current session.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries to store.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO message_groups (id) VALUES (?)",
            (self.session_id,),
        )

        cursor.execute(
            "SELECT MAX(group_id) FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        max_group_id = cursor.fetchone()[0]
        group_id = 0 if max_group_id is None else max_group_id + 1

        for message in messages:
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, group_id) VALUES (?, ?, ?, ?)
                """,
                (self.session_id, message["role"], message["content"], group_id),
            )
        self.conn.commit()

    def patch_create(self, client: "openai.Client") -> None:
        """
        Patch the OpenAI client's create method to store messages in the database.

        Args:
            client (openai.Client): The OpenAI client instance to patch.
        """
        self._client = client
        self._original_create = client.chat.completions.create

        async def stream_response(*args: Any, **kwargs: Any):
            content = ""
            role = ""
            messages = kwargs.get("messages", [])
            async for chunk in await self._original_create(*args, **kwargs):
                yield chunk
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        content += delta.content
                    if delta.role:
                        role = delta.role
            if content and role:
                messages.append({"role": role, "content": content})
            self.store_messages(messages)

        async def non_stream_response(*args: Any, **kwargs: Any):
            response = await self._original_create(*args, **kwargs)
            messages = kwargs.get("messages", [])
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
                role = response.choices[0].message.role
                messages.append({"role": role, "content": content})
            self.store_messages(messages)

        async def patched_async_create(*args: Any, **kwargs: Any) -> Any:
            stream = kwargs.get("stream", False)
            if stream:
                return stream_response(*args, **kwargs)
            else:
                return await non_stream_response(*args, **kwargs)

            return response

        self._client.chat.completions.create = patched_async_create

    def get_session_message_groups(
        self, session_id: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the groups of messages from the last session, or a specific session if provided.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing group_id and messages for each group.
        """
        cursor = self.conn.cursor()

        if session_id is None:
            cursor.execute(
                "SELECT id FROM message_groups ORDER BY timestamp DESC LIMIT 1"
            )
            session_id = cursor.fetchone()[0]

        cursor.execute(
            "SELECT DISTINCT group_id FROM messages WHERE session_id = ? ORDER BY group_id",
            (session_id,),
        )
        group_ids = cursor.fetchall()

        groups = []
        for (group_id,) in group_ids:
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND group_id = ? ORDER BY timestamp",
                (session_id, group_id),
            )
            messages = cursor.fetchall()
            group_messages = [
                {"role": role, "content": content} for role, content in messages
            ]
            groups.append({"group_id": group_id, "messages": group_messages})

        return groups

    def get_all_session_message_groups(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the groups of messages from all sessions.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing group_id and messages for each group.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM messages")
        session_ids = cursor.fetchall()

        all_groups = {}
        for (session_id,) in session_ids:
            all_groups[session_id] = self.get_session_message_groups(session_id)

        return all_groups

    def __del__(self) -> None:
        """Close the database connection and reverts the client create when the object is deleted."""
        self._client.chat.completions.create = self._original_create
        self.conn.close()

    def close(self) -> None:
        """Close the database connection and reverts the client create."""
        self._client.chat.completions.create = self._original_create
        self.conn.close()
