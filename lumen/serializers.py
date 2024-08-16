import uuid

from io import StringIO

import numpy as np
import pandas as pd

from .base import MultiTypeComponent
from .config import SessionCache


class Serializer(MultiTypeComponent):

    serializer_type = None

    @classmethod
    def serialize(self, data) -> dict[str, any]:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data) -> dict[str, any]:
        serialize_spec = dict(data.pop('serializer', {}))
        serializer = cls.from_spec(serialize_spec)
        return serializer.deserialize(data)


class CSVSerializer(Serializer):

    serializer_type = 'csv'

    def serialize(self, data) -> dict[str, any]:
        csv = StringIO()
        index = list(data.index.names)
        if len(index) == 1 and index[0] is None:
            index = False
        data.to_csv(csv, index=bool(index))
        csv.seek(0)
        return {
            'data': csv.read(),
            'type': self.serializer_type,
            'index': index,
            'date_cols': list(data.select_dtypes(np.datetime64).columns),
            'dtypes': {col: str(data[col].dtype) for col in data.columns},
            'serializer': self.to_spec()
        }

    def deserialize(self, data) -> dict[str, any]:
        data = StringIO(data['data'])
        df = pd.read_csv(
            data, parse_dates=data['date_cols']
        ).astype(data['dtypes'])
        index_cols = data['index']
        if index_cols:
            df = df.set_index(index_cols)
        return df


class SessionSerializer(Serializer):

    serializer_type = 'session'

    _session_cache = SessionCache()

    def serialize(self, data) -> dict[str, any]:
        data_id = uuid.uuid4().hex
        self._session_cache[data_id] = data
        return {
            'type': self.serializer_type,
            'id': data_id,
            'serializer': self.to_spec()
        }

    def deserialize(self, data) -> dict[str, any]:
        if data['id'] in self._session_cache:
            return self._session_cache[data['id']]
        else:
            raise KeyError(
                'Data was not found in session cache. Ensure you are'
                'loading a dataset that was created in the same session '
                'you are attempting to load it from.'
            )
