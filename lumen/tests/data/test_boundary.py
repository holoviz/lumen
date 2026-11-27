"""
lumen.data must stay importable and usable without any LLM library and
without lumen.ai, whose package __init__ loads the agents, LLM wrappers and UI.
"""
import subprocess
import sys
import textwrap

BLOCKED = ("openai", "instructor", "anthropic", "pydantic_ai", "mistralai", "lumen.ai")

SCRIPT = textwrap.dedent("""
    import importlib.abc
    import pkgutil
    import sys

    BLOCKED = {blocked!r}

    def blocked(name):
        return any(name == b or name.startswith(b + ".") for b in BLOCKED)

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if blocked(name):
                raise ImportError(f"{{name}} is blocked")
            return None

    sys.meta_path.insert(0, Blocker())

    import lumen.data

    for info in pkgutil.iter_modules(lumen.data.__path__, "lumen.data."):
        importlib.import_module(info.name)

    from lumen.config import SOURCE_TABLE_SEPARATOR as SEP
    from lumen.data import (
        Column, Metaset, TableCatalogEntry, build_distinct_values_sql,
        clean_sql, validate_read_only_sql,
    )

    validate_read_only_sql(clean_sql("SELECT a FROM t"), "duckdb")
    build_distinct_values_sql("SELECT * FROM t", "a", "duckdb", like="x")
    slug = f"s{{SEP}}t"
    Metaset(query=None, catalog={{slug: TableCatalogEntry(slug, 1.0, [Column("a")])}}).compact_context()

    loaded = sorted(name for name in sys.modules if blocked(name))
    assert not loaded, loaded
    print("ok")
""").format(blocked=BLOCKED)


def test_lumen_data_imports_no_llm_library_or_lumen_ai():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
