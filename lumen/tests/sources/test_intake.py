import os
import yaml

import pandas as pd

from lumen.sources.intake import IntakeSource


def test_intake_source_from_file():
    root = os.path.dirname(__file__)
    source = IntakeSource(uri=os.path.join(root, 'catalog.yml'),
                          root=root)
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test'), df)


def test_intake_source_from_dict():
    root = os.path.dirname(__file__)
    with open(os.path.join(root, 'catalog.yml')) as f:
        catalog = yaml.load(f, Loader=yaml.Loader)
    source = IntakeSource(catalog=catalog, root=root)
    df = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(source.get('test'), df)
