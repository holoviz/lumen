import datetime
import json
import sqlite3

from abc import abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any

import param

from pydantic import BaseModel


class Interceptor(param.Parameterized):

    db_path = param.String(
        default="messages.db", doc="Path to the SQLite database file"
    )

    def __init__(self, **params):
        super().__init__(**params)
        needs_init = not Path(self.db_path).exists()
        self.conn = self._create_connection()
        if needs_init:
            self.init_db()
        self._client = self._original_create = self._original_create_response = None
        self._last_batch_id = None
        self.session_id = self._generate_session_id()

    def _create_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        first_message_timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"conv_{first_message_timestamp}"

    def _dump_response_model(self, response_model: BaseModel) -> str:
        """Dump the response model to a JSON string."""
        # using json.dumps instead of model_dump_json
        # for more consistent serialization
        content = json.dumps(response_model.model_dump())
        return content

    def _select_max_batch_id(self) -> int:
        """Get the maximum batch ID for the current session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(batch_id) FROM messages WHERE session_id = ?",
            (self.session_id,),
        )
        max_batch_id = cursor.fetchone()[0]
        return 0 if max_batch_id is None else max_batch_id

    @abstractmethod
    def patch_client(self, client) -> None:
        """
        Patch the LLM client's create method to store messages and arguments in the database.

        Args:
            client: The LLM client instance to patch.
        """

    def init_db(self) -> None:
        """Initialize the database by creating necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS message_batches (
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
                batch_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES message_batches(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS message_kwargs (
                session_id TEXT,
                batch_id INTEGER,
                key TEXT,
                value TEXT,
                FOREIGN KEY (session_id) REFERENCES message_batches(id),
                FOREIGN KEY (batch_id) REFERENCES messages(batch_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                session_id TEXT,
                batch_id INTEGER,
                content TEXT,
                FOREIGN KEY (session_id) REFERENCES message_batches(id),
                FOREIGN KEY (batch_id) REFERENCES messages(batch_id)
            )
            """
        )
        self.conn.commit()

    def store_messages(
        self, messages: list[dict[str, str]], **kwargs: dict[str, Any]
    ) -> None:
        """
        Store messages and keyword arguments in the database for the current session.

        Args:
            messages: List of message dictionaries to store.
            kwargs: The keyword arguments passed to the create method.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO message_batches (id) VALUES (?)",
            (self.session_id,),
        )

        batch_id = self._select_max_batch_id() + 1

        for message in messages:
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, batch_id) VALUES (?, ?, ?, ?)
                """,
                (self.session_id, message["role"], message["content"], batch_id),
            )

        for key, value in kwargs.items():
            if key == "messages":
                continue
            cursor.execute(
                """
                INSERT INTO message_kwargs (session_id, batch_id, key, value) VALUES (?, ?, ?, ?)
                """,
                (self.session_id, batch_id, key, json.dumps(value)),
            )

        self.conn.commit()

        self._last_batch_id = batch_id

    def store_response(self, content: str) -> None:
        cursor = self.conn.cursor()

        print(f"Storing response for batch {self._last_batch_id} {content}")
        # Store the response based on the last batch_id
        cursor.execute(
            """
            INSERT INTO responses (session_id, batch_id, content)
            VALUES (?, ?, ?)
            """,
            (self.session_id, self._last_batch_id, content),
        )
        self.conn.commit()

    def get_session(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """
        Retrieve the session batches of inputs from the last session, or a specific session if provided.

        Args:
            session_id: The session ID to retrieve batches from. If not provided, the last session is used.

        Returns:
            A list of dictionaries containing batch_id, messages, and kwargs for each batch.
        """
        cursor = self.conn.cursor()

        if session_id is None:
            cursor.execute(
                "SELECT id FROM message_batches ORDER BY timestamp DESC LIMIT 1"
            )
            session_id = cursor.fetchone()[0]

        cursor.execute(
            "SELECT DISTINCT batch_id FROM messages WHERE session_id = ? ORDER BY batch_id",
            (session_id,),
        )
        batch_ids = cursor.fetchall()

        batches = []
        for (batch_id,) in batch_ids:
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND batch_id = ? ORDER BY timestamp",
                (session_id, batch_id),
            )
            messages = cursor.fetchall()
            batch_messages = [
                {"role": role, "content": content} for role, content in messages
            ]

            cursor.execute(
                """
                SELECT key, value
                FROM message_kwargs
                WHERE session_id = ? AND batch_id = ?
                """,
                (session_id, batch_id),
            )
            kwargs_data = cursor.fetchall()
            kwargs_dict = {key: json.loads(value) for key, value in kwargs_data}

            cursor.execute(
                """
                SELECT content
                FROM responses
                WHERE session_id = ? AND batch_id = ?
                """,
                (session_id, batch_id),
            )
            response_data = cursor.fetchone()
            if response_data is None:
                response_content = None
            else:
                response_content = response_data[0]

            batches.append(
                {
                    "batch_id": batch_id,
                    "messages": batch_messages,
                    "kwargs": kwargs_dict,
                    "response": response_content,
                }
            )

        return batches

    def get_all_sessions(self) -> dict[str, dict[str, Any]]:
        """
        Retrieve the batches of messages from all sessions.

        Returns:
            A dictionary containing session_id as keys and the corresponding
                list of message batches for each session.
        """
        all_batches = {}
        for session_id in self.get_session_ids():
            all_batches[session_id] = self.get_session(session_id)

        return all_batches

    def get_session_ids(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM message_batches")
        return [row[0] for row in cursor.fetchall()]

    def unpatch(self) -> None:
        """Close the database connection and reverts the client create."""
        if self._original_create_response is not None:
            self._client.chat.completions.create = self._original_create_response
        if self._original_create is not None:
            self._client.chat.completions.create = self._original_create
        self.conn.close()

    def __del__(self) -> None:
        """Close the database connection and reverts the client create when the object is deleted."""
        self.unpatch()


class OpenAIInterceptor(Interceptor):

    def patch_client(self, client) -> None:
        """
        Patch the OpenAI client's create method to store messages and arguments in the database.

        Args:
            client: The OpenAI client instance to patch.
        """
        self._client = client
        self._original_create = client.chat.completions.create

        async def stream_response(*args: Any, **kwargs: Any):
            async for chunk in await self._original_create(*args, **kwargs):
                yield chunk
            self.store_messages(**kwargs)

        async def non_stream_response(*args: Any, **kwargs: Any):
            response = await self._original_create(*args, **kwargs)
            self.store_messages(**kwargs)
            return response

        @wraps(client.chat.completions.create)
        async def patched_async_create(*args: Any, **kwargs: Any) -> Any:
            stream = kwargs.get("stream", False)
            if stream:
                return stream_response(*args, **kwargs)
            else:
                return await non_stream_response(*args, **kwargs)

        self._client.chat.completions.create = patched_async_create

    def patch_client_response(self, client) -> None:
        """
        Patch the OpenAI client's create method to store the response in the database.

        Args:
            client: The OpenAI client instance to patch.
        """
        self._client = client
        self._original_create_response = client.chat.completions.create

        async def stream_response(*args: Any, **kwargs: Any):
            content = ""
            response = await self._original_create_response(*args, **kwargs)
            async for chunk in response:
                yield chunk
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        content += delta.content

            print(content, "CONTENT")
            # last chunk contains all the content
            if not content and isinstance(chunk, BaseModel):
                content = self._dump_response_model(chunk)

            if content:
                self.store_response(content)
            else:
                print("Could not intercept a response")

        async def non_stream_response(*args: Any, **kwargs: Any):
            content = ""
            response = await self._original_create_response(*args, **kwargs)
            if hasattr(response, "choices"):
                content = response.choices[0].message.content

            print(content, "CONTENT")
            if not content and isinstance(response, BaseModel):
                content = self._dump_response_model(response)

            if content:
                self.store_response(content)
            else:
                print("Could not intercept a response.")

            return response

        @wraps(client.chat.completions.create)
        async def patched_create_response(*args: Any, **kwargs: Any) -> Any:
            stream = kwargs.get("stream", False)
            if stream:
                return stream_response(*args, **kwargs)
            else:
                return await non_stream_response(*args, **kwargs)

        client.chat.completions.create = patched_create_response
