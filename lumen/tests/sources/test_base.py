import os

from lumen.config import config
from lumen.sources import FileSource, Source
from lumen.state import state


def test_source_resolve_module_type():
    assert Source._get_type('lumen.sources.base.Source') is Source
    assert Source.source_type is None


def test_source_from_spec_in_state():
    config_root = config._root
    root = os.path.dirname(__file__)
    config._root = root
    original_source = FileSource(tables={'test': 'test.csv'},  kwargs={'parse_dates': ['D']})
    state.sources['original'] = original_source
    from_spec_source = Source.from_spec('original')

    # source loaded from spec is identical with the original source
    assert type(from_spec_source) == type(original_source)
    assert from_spec_source.name == original_source.name
    assert from_spec_source.cache_dir == original_source.cache_dir
    assert from_spec_source.shared == original_source.shared
    assert from_spec_source.root == original_source.root
    assert from_spec_source.source_type == original_source.source_type
    assert from_spec_source._supports_sql == original_source._supports_sql

    # clean up cache memory
    config._root = config_root
    for source in state.global_sources.values():
        source.clear_cache()
    state.global_sources.clear()
