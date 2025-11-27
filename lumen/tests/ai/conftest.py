import sys

import param
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from pydantic import BaseModel

from lumen.ai.llm import Llm, Message
from lumen.sources.duckdb import DuckDBSource


def pytest_collection_modifyitems(config, items):
    if not sys.platform.startswith("win"):
        return

    for item in items:
        if "duckdb" in item.nodeid.lower() or any("duckdb" in fn for fn in item.fixturenames):
            item.add_marker(pytest.mark.xdist_group("duckdb"))


class MockLLM(Llm):

    model_kwargs = param.Dict({'default': 'mock'})

    _supports_stream = False
    _supports_model_stream = False

    async def invoke(
        self,
        messages: list[Message],
        system: str = "",
        response_model: BaseModel | None = None,
        allow_partial: bool = False,
        model_spec: str | dict = "default",
        **input_kwargs,
    ) -> BaseModel:
        ret = self._responses[self._index]
        if callable(ret):
            ret = ret()
        self._index += 1
        return ret

    def set_responses(self, responses):
        self._responses = responses
        self._index = 0


@pytest.fixture
def llm():
    return MockLLM()


@pytest.fixture
def tiny_source():
    return DuckDBSource(tables={
    'tiny': """
        SELECT * FROM (
          VALUES (1,'A',10.5),
                 (2,'B',20.0),
                 (3,'A', 7.5)
        ) AS t(id, category, value)
    """
    })
