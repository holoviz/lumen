import os
import tempfile

from unittest.mock import Mock

import pandas as pd
import panel as pn
import pytest

from bokeh.document import Document

from lumen.config import config
from lumen.sources import FileSource, Source
from lumen.state import state
from lumen.variables import Variables


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
def mixed_df():
    yield pd._testing.makeMixedDataFrame()

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
    state._spec.clear()
    state._loading.clear()
    state._sources.clear()
    state._filters.clear()
    state._pipelines.clear()
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


@pytest.fixture
def expected_filtered_df(source_tables, table_column_value_type):
    table, column, value, type = table_column_value_type
    df = source_tables[table]

    if type == 'single_value':
        if value is None:
            df = df[df[column].isnull()]
        else:
            df = df[df[column] == value]

    elif type == 'range':
        begin, end = value
        df = df[(df[column] >= begin) & (df[column] <= end)]

    elif type == 'range_list':
        conditions = False
        for range in value:
            begin, end = range
            conditions |= ((df[column] >= begin) & (df[column] <= end))
        df = df[conditions]

    elif type == 'list':
        df = df[df[column].isin(value)]

    elif type == 'date':
        df = df[df[column] == pd.to_datetime(value)]

    elif type == 'date_range':
        begin, end = value
        df = df[(df[column] >= pd.to_datetime(begin)) & (df[column] <= pd.to_datetime(end))]

    return df

@pytest.fixture
def penguins_file(tmp_path):
    # created with
    # df = pd.read_csv(url).sample(5).reset_index(drop=True).to_csv()
    fn = tmp_path / "penguins.csv"
    penguins = """
    ,species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year
    0,Gentoo,Biscoe,46.7,15.3,219.0,5200.0,male,2007
    1,Adelie,Dream,42.2,18.5,180.0,3550.0,female,2007
    2,Gentoo,Biscoe,51.1,16.3,220.0,6000.0,male,2008
    3,Adelie,Biscoe,45.6,20.3,191.0,4600.0,male,2009
    4,Adelie,Torgersen,37.8,17.1,186.0,3300.0,,2007
    """
    fn.write_text(penguins)
    return fn
