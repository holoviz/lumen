import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.tools.metaset_docs_llm_tools import (
    make_load_metaset_relevant_docs_tool,
)
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.data.catalog import DocumentChunk, Metaset, TableCatalogEntry


def _metaset(docs=None):
    slug = f"S{SOURCE_TABLE_SEPARATOR}t"
    entry = TableCatalogEntry(table_slug=slug, similarity=1.0, columns=[])
    return Metaset(query=None, catalog={slug: entry}, docs=docs)


def test_load_metaset_relevant_docs_tool():
    ms = _metaset(docs=[DocumentChunk(filename="g.md", text="hello world", similarity=0.77)])
    tools = make_load_metaset_relevant_docs_tool({"metaset": ms})
    assert len(tools) == 1
    tool = tools[0]
    assert "0.770" in tool.purpose or "0.77" in tool.purpose
    assert "chunk(s)" in tool.purpose
    out = tool.function(max_chunks=5)
    assert "hello world" in out
    assert "similarity range" in out.lower()
    assert not make_load_metaset_relevant_docs_tool({"metaset": _metaset()})
