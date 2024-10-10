import datetime
import json
import sqlite3
import uuid

from abc import abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import param

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Invocation(BaseModel):
    input_id: int
    prompt: str
    messages: list[Message]
    response: str | None
    kwargs: dict[str, Any]
    invocation_id: str

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize messages into a list of dictionaries."""
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]


class Session(BaseModel):
    session_id: str
    invocations: list[Invocation]


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
        self._last_invocation_id = None
        self.session_id = self._generate_session_id()

    def _create_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        first_message_timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"session_{first_message_timestamp}"

    def _dump_response_model(self, response_model: BaseModel) -> str:
        """Dump the response model to a JSON string."""
        return json.dumps(response_model.model_dump())

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
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS invocations (
                invocation_id TEXT PRIMARY KEY,
                session_id TEXT,
                messages TEXT,
                message_kwargs TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
            """
        )
        self.conn.commit()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                invocation_id INTEGER,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (invocation_id) REFERENCES invocations(invocation_id)
            )
            """
        )

    def reset_db(self) -> None:
        """Reset the database by deleting all tables."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS invocations")
        cursor.execute("DROP TABLE IF EXISTS sessions")
        self.conn.commit()
        self.init_db()

    def delete_session(self, session_id: str | None = None) -> None:
        """Delete the last session from the database."""
        cursor = self.conn.cursor()
        if session_id is None:
            session_id = self.get_session_ids()[-1]
        cursor.execute("DELETE FROM invocations WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def store_invocation(
        self, messages: list[dict[str, str]], **kwargs: dict[str, Any]
    ) -> None:
        """
        Store messages and keyword arguments in the database for the current session.

        Args:
            messages: List of message dictionaries to store.
            kwargs: The keyword arguments passed to the create method.
        """
        cursor = self.conn.cursor()
        self._last_invocation_id = invocation_id = uuid.uuid4().hex

        cursor.execute(
            """
            INSERT INTO invocations (invocation_id, session_id, messages, message_kwargs)
            VALUES (?, ?, ?, ?)
            """,
            (invocation_id, self.session_id, json.dumps(messages), json.dumps(kwargs)),
        )
        self.conn.commit()

    def store_response(self, content: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO responses (invocation_id, content, timestamp)
            VALUES (?, ?, ?)
            """,
            (self._last_invocation_id, content, datetime.datetime.now()),
        )
        self.conn.commit()

    def get_session(self, session_id: str | None = None) -> Session:
        """
        Retrieve the session invocations of inputs from the last session, or a specific session if provided.

        Args:
            session_id: The session ID to retrieve invocations from. If not provided, the last session is used.

        Returns:
            A Session object containing invocations for the session.
        """
        cursor = self.conn.cursor()

        if session_id is None:
            cursor.execute(
                "SELECT session_id FROM invocations ORDER BY timestamp DESC LIMIT 1"
            )
            try:
                session_id = cursor.fetchone()[0]
            except TypeError:
                return Session(session_id="", invocations=[])

        cursor.execute(
            """
            SELECT
                invocations.invocation_id,
                messages,
                message_kwargs,
                responses.content AS response
            FROM invocations
            LEFT JOIN responses ON invocations.invocation_id = responses.invocation_id
            WHERE invocations.session_id = ?
            ORDER BY invocations.timestamp ASC
            """,
            (session_id,),
        )
        invocation_data = cursor.fetchall()

        input_id = -1
        invocations = []
        prev_user_content = None
        for invocation_id, invocation_messages, message_kwargs, response in invocation_data:
            messages = []
            for message in json.loads(invocation_messages):
                if message["role"] == "user":
                    user_content = message["content"]
                    if prev_user_content != user_content:
                        prev_user_content = user_content
                        input_id += 1
                messages.append(Message(role=message["role"], content=message["content"]))

            invocations.append(
                Invocation(
                    prompt=user_content,
                    invocation_id=invocation_id,
                    input_id=input_id,
                    messages=messages,
                    kwargs=json.loads(message_kwargs),
                    response=response,
                )
            )

        return Session(session_id=session_id, invocations=invocations)

    def get_all_sessions(self) -> dict[str, Session]:
        """
        Retrieve the invocations of messages from all sessions.

        Returns:
            A dictionary containing session_id as keys and the corresponding
                Session object for each session.
        """
        all_sessions = {}
        for session_id in self.get_session_ids():
            all_sessions[session_id] = self.get_session(session_id)

        return all_sessions

    def get_session_ids(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM invocations")
        return [row[0] for row in cursor.fetchall()]

    def unpatch(self) -> None:
        """Close the database connection and revert the client create."""
        if self._original_create_response is not None:
            self._client.chat.completions.create = self._original_create_response
        if self._original_create is not None:
            self._client.chat.completions.create = self._original_create
        self.conn.close()

    def __del__(self) -> None:
        """Close the database connection and revert the client create when the object is deleted."""
        self.unpatch()


class OpenAIInterceptor(Interceptor):

    def patch_client(
        self, client, mode: Literal["store_all", "store_inputs"] = "store_all"
    ) -> None:
        """
        Patch the OpenAI client's create method to store messages and arguments in the database.

        Args:
            client: The OpenAI client instance to patch.
            mode: The mode to patch the client in.
                If "store_all", responses are generated, and everything is stored.
                If "store_inputs", responses are generated, but only input arguments are stored.
        """
        self._client = client
        self._original_create = client.chat.completions.create

        async def stream_response(*args: Any, **kwargs: Any):
            async for chunk in await self._original_create(*args, **kwargs):
                yield chunk
            self.store_invocation(**kwargs)

        async def non_stream_response(*args: Any, **kwargs: Any):
            response = await self._original_create(*args, **kwargs)
            self.store_invocation(**kwargs)
            return response

        @wraps(client.chat.completions.create)
        async def patched_async_create(*args: Any, **kwargs: Any) -> Any:
            stream = kwargs.get("stream", False)
            if stream:
                return stream_response(*args, **kwargs)
            else:
                return await non_stream_response(*args, **kwargs)

        self._client.chat.completions.create = patched_async_create
        if mode == "store_all":
            self.patch_client_response(client)

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
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        content += delta.content

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
