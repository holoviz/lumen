import pytest

from lumen.config import config
from lumen.sources import FileSource
from lumen.state import state


@pytest.fixture
def set_root():
    root = config._root
    def _set_root(root):
        config._root = root
    yield _set_root
    config._root = root


@pytest.fixture
def make_filesource():
    root = config._root
    def create(root):
        config._root = root
        source = FileSource(tables={'test': 'test.csv'},  kwargs={'parse_dates': ['D']})
        state.sources['original'] = source
        return source
    yield create
    config._root = root
    state.global_sources.clear()
