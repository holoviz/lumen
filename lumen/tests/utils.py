import pandas as pd


def assert_frame_equal_no_dtype_check(left, right, **kwargs):
    return pd.testing.assert_frame_equal(left, right, check_dtype=False, **kwargs)
