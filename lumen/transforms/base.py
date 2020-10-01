"""
A Transform component is given the data for a particular metric and
transforms it in some arbitrary way.
"""

import datetime as dt

import pandas as pd
import param


class Transform(param.Parameterized):
    """
    A Transform provides the ability to transform the metric data
    supplied by a QueryAdaptor.
    """

    transform_type = None

    __abstract = True

    @classmethod
    def get(cls, transform_type):
        for transform in param.concrete_descendents(cls).values():
            if transform.transform_type == transform_type:
                return transform
        raise ValueError(f"No Transform for transform_type '{transform_type}' could be found.")

    def apply(self, metric_data):
        """
        Given some metric_data transform it in some way and return it.

        Parameters
        ----------
        metric_data : pandas.DataFrame
            The queried metric data as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the transformed metric data.
        """
        return metric_data


class HistoryTransform(Transform):
    """
    The HistoryTransform accumulates a history of the queried metric
    data in a buffer up to the supplied length and (optionally) adds
    a date_column to the data.
    """

    date_column = param.String(doc="""
        If defined adds a date column with the supplied name.""")

    length = param.Integer(default=10, bounds=(1, None), doc="""
        Accumulates a history of data.""")

    transform_type = 'history'

    def __init__(self, **params):
        super().__init__(**params)
        self._buffer = []

    def apply(self, metric_data):
        """
        Accumulates a history of the metric data in a buffer up to
        the declared `length` and optionally adds the current datetime
        to the declared `date_column`.

        Parameters
        ----------
        metric_data : pandas.DataFrame
            The queried metric data as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the buffered history of the metric
            data.
        """
        if self.date_column:
            metric_data = metric_data.copy()
            metric_data[self.date_column] = dt.datetime.now()
        self._buffer.append(metric_data)
        self._buffer[:] = self._buffer[-self.length:]
        return pd.concat(self._buffer)
