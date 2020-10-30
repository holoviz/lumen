"""
The Transform components allow transforming tables in arbitrary ways.
"""

import datetime as dt

import pandas as pd
import param


class Transform(param.Parameterized):
    """
    A Transform provides the ability to transform a table supplied by
    a Source.
    """

    transform_type = None

    __abstract = True

    @classmethod
    def _get_type(cls, transform_type):
        try:
            __import__(f'lumen.transforms.{transform_type}')
        except Exception:
            pass
        for transform in param.concrete_descendents(cls).values():
            if transform.transform_type == transform_type:
                return transform
        raise ValueError(f"No Transform for transform_type '{transform_type}' could be found.")

    @classmethod
    def from_spec(cls, spec):
        """
        Resolves a Transform specification.

        Parameters
        ----------
        spec: dict
            Specification declared as a dictionary of parameter values.

        Returns
        -------
        The resolved Transform object.
        """
        spec = dict(spec)
        transform_type = Transform._get_type(spec.pop('type', None))
        return transform_type(**transform)

    def apply(self, table):
        """
        Given a table transform it in some way and return it.

        Parameters
        ----------
        table : DataFrame
            The queried table as a DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the transformed data.
        """
        return table


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

    def apply(self, table):
        """
        Accumulates a history of the data in a buffer up to the
        declared `length` and optionally adds the current datetime to
        the declared `date_column`.

        Parameters
        ----------
        data : DataFrame
            The queried table as a DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the buffered history of the data.
        """
        if self.date_column:
            table = table.copy()
            table[self.date_column] = dt.datetime.now()
        self._buffer.append(table)
        self._buffer[:] = self._buffer[-self.length:]
        return pd.concat(self._buffer)
