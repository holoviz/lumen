"""
A Transform component is given the data for a particular variable and
transforms it.
"""

import datetime as dt

import pandas as pd
import param


class Transform(param.Parameterized):
    """
    A Transform provides the ability to transform the data supplied by
    a Source.
    """

    transform_type = None

    __abstract = True

    @classmethod
    def _get_type(cls, transform_type):
        for transform in param.concrete_descendents(cls).values():
            if transform.transform_type == transform_type:
                return transform
        raise ValueError(f"No Transform for transform_type '{transform_type}' could be found.")

    def apply(self, data):
        """
        Given some data transform it in some way and return it.

        Parameters
        ----------
        data : pandas.DataFrame
            The queried data as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the transformed data.
        """
        return data


class HistoryTransform(Transform):
    """
    The HistoryTransform accumulates a history of the queried data in
    a buffer up to the supplied length and (optionally) adds a
    date_column to the data.
    """

    date_column = param.String(doc="""
        If defined adds a date column with the supplied name.""")

    length = param.Integer(default=10, bounds=(1, None), doc="""
        Accumulates a history of data.""")

    transform_type = 'history'

    def __init__(self, **params):
        super().__init__(**params)
        self._buffer = []

    def apply(self, data):
        """
        Accumulates a history of the data in a buffer up to the
        declared `length` and optionally adds the current datetime to
        the declared `date_column`.

        Parameters
        ----------
        data : pandas.DataFrame
            The queried data as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the buffered history of the data.
        """
        if self.date_column:
            data = data.copy()
            data[self.date_column] = dt.datetime.now()
        self._buffer.append(data)
        self._buffer[:] = self._buffer[-self.length:]
        return pd.concat(self._buffer)
