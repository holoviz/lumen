import os
import tempfile

import pytest

import panel as pn

from bokeh.document import Document

from lumen.config import config
from lumen.sources import FileSource, Source
from lumen.state import state
from lumen.variables import Variables

from unittest.mock import Mock

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
    def create(root, **kwargs):
        config._root = root
        source = FileSource(tables={'test': 'test.csv'},  kwargs={'parse_dates': ['D']}, **kwargs)
        state.sources['original'] = source
        return source
    yield create
    config._root = root
    for source in state.global_sources.values():
        source.clear_cache()
    state.global_sources.clear()


@pytest.fixture
def make_variable_filesource():
    root = config._root
    def create(root, **kwargs):
        config._root = root
        state._variables[None] = Variables.from_spec({'tables': {'type': 'constant', 'default': {'test': 'test.csv'}}})
        source = Source.from_spec(dict({
            'type': 'file',
            'tables': '$variables.tables',
            'kwargs': {'parse_dates': ['D']}
        }))
        state.sources['original'] = source
        return source
    yield create
    config._root = root
    for source in state.global_sources.values():
        source.clear_cache()
    state.global_sources.clear()
    state._variables.clear()

@pytest.fixture
def yaml_file():
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yield tf
    tf.close()
    os.unlink(tf.name)


@pytest.fixture
def state_userinfo():
    mock_state = Mock()
    mock_state.user_info = {'email': 'lumen@holoviz.org', 'user': 'lumen'}
    mock_state.user = 'lumen'
    state = pn.state
    pn.state = mock_state
    yield
    pn.state = state


@pytest.fixture(autouse=True)
def clear_state():
    "Clear the state after the execution of each test"
    yield
    state.global_sources.clear()
    state.global_filters.clear()
    state.spec.clear()
    state._loading.clear()
    state._sources.clear()
    state._filters.clear()
    state._variables.clear()


@pytest.fixture
def document():
    doc = Document()
    with pn.io.server.set_curdoc(doc):
        yield

@pytest.fixture
def cachedir():
    tmp_dir = tempfile.TemporaryDirectory()
    yield tmp_dir.name
    tmp_dir.cleanup()
