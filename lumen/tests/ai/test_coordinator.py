import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.coordinator import Planner
from lumen.ai.tools import IterativeTableLookup, TableLookup
from lumen.ai.vector_store import NumpyVectorStore


async def test_planner_instantiate():
    Planner()


async def test_planner_instantiate_tools_shared_vector_store():
    planner = Planner()

    assert len(planner._tools['main']) == 2
    tool1, tool2 = planner._tools['main']
    assert isinstance(tool1, TableLookup)
    assert isinstance(tool2, IterativeTableLookup)

    assert tool1.vector_store is tool2.vector_store


async def test_planner_instantiate_tools_provided_vector_store():
    vector_store = NumpyVectorStore()
    planner = Planner(vector_store=vector_store)

    assert len(planner._tools['main']) == 2
    tool1, tool2 = planner._tools['main']
    assert isinstance(tool1, TableLookup)
    assert isinstance(tool2, IterativeTableLookup)

    assert tool1.vector_store is vector_store
    assert tool2.vector_store is vector_store
