import os

import pandas as pd
import pytest

import yaml

from lumen.sources.intake import IntakeSource


@pytest.fixture
def source():
    root = os.path.dirname(__file__)
    return IntakeSource(
        uri=os.path.join(root, 'catalog.yml'), root=root
    )


def test_intake_resolve_module_type():
    assert IntakeSource._get_type('lumen.sources.intake_sql.IntakeSource') is IntakeSource
    assert IntakeSource.source_type == 'intake'


def test_intake_source_from_file(source):
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test'), df)


def test_intake_source_from_dict():
    root = os.path.dirname(__file__)
    with open(os.path.join(root, 'catalog.yml')) as f:
        catalog = yaml.load(f, Loader=yaml.Loader)
    source = IntakeSource(catalog=catalog, root=root)
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test'), df)
