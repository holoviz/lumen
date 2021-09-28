import os
import tempfile

import pytest

import panel as pn

from lumen.config import config
from lumen.sources import FileSource
from lumen.state import state

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
    def create(root):
        config._root = root
        source = FileSource(tables={'test': 'test.csv'},  kwargs={'parse_dates': ['D']})
        state.sources['original'] = source
        return source
    yield create
    config._root = root
    state.global_sources.clear()



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
