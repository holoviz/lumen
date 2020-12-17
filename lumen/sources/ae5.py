import datetime as dt

import param
import pandas as pd
import requests

from ae5_tools.api import AEUserSession

from .base import Source, cached
from ..util import get_dataframe_schema


class AE5Source(Source):

    hostname = param.String(doc="URL of the AE5 host.")

    username = param.String(doc="Username to authenticate with AE5.")

    password = param.String(doc="Password to authenticate with AE5.")

    k8s_endpoint = param.String(default='k8s')

    pool_size = param.Integer(default=100, doc="""
        Size of HTTP socket pool.""")

    source_type = 'ae5'

    _deployment_columns = [
        'id', 'name', 'url', 'owner', 'resource_profile', 'public', 'state'
    ]

    _session_columns = [
        'id', 'name', 'url', 'owner', 'resource_profile', 'state'
    ]

    _units = {
        'm': 0.001,
        None: 1,
        'Ki': 1024,
        'Mi': 1024**2,
        'Gi': 1024**3,
        'Ti': 1024**4
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._session = AEUserSession(
            self.hostname, self.username, self.password, persist=False,
            k8s_endpoint=self.k8s_endpoint
        )
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=self.pool_size)
        self._session.session.mount('https://', adapter)
        self._session.session.mount('http://', adapter)

    @classmethod
    def _convert_value(cls, value):
        if value == '0':
            return 0
        val_unit = [m for m in cls._units if m is not None and value.endswith(m)]
        val_unit = val_unit[0] if val_unit else None
        if not val_unit:
            return float(value)
        scale_factor = cls._units[val_unit]
        return float(value[:-len(val_unit)]) * scale_factor

    def _get_record(self, deployment):
        container_info = deployment.get('_k8s', {}).get('containers', {})
        record = {
            "id": deployment['id'],
            "name": deployment['name'],
            "url": deployment['url'],
            "resource_profile": deployment['resource_profile']
        }

        if 'app' not in container_info:
            nan = float('NaN')
            return dict(record, cpu=nan, memory=nan, restarts=nan, uptime='-')

        container = container_info['app']
        limits = container['limits']
        usage = container['usage']

        # CPU Usage
        cpu_cap = self._convert_value(limits['cpu'])
        if 'cpu' in usage:
            cpu = self._convert_value(usage['cpu'])
            record['cpu'] = round((cpu / cpu_cap)*100, 2)
        else:
            record['cpu'] = float('nan')

        # Memory usage
        mem_cap = self._convert_value(limits['memory'])
        if 'memory' in usage:
            mem = self._convert_value(usage['memory'])
            record['memory'] = round((mem / mem_cap)*100, 2)
        else:
            record['memory'] = float('nan')

        # Uptime
        started = container.get('since')
        record["restarts"] = container['restarts']
        if started:
            started_dt = dt.datetime.fromisoformat(started.replace('Z', ''))
            uptime = str(dt.datetime.now()-started_dt)
            record["uptime"] = uptime[:uptime.index('.')]
        else:
            record["uptime"] = '-'

        return record

    def get_schema(self, table=None):
        schemas = {}
        if table in (None, 'deployments'):
            deployments = self.get('deployments')
            schemas['deployments'] = get_dataframe_schema(deployments)['items']['properties']
        if table in (None, 'sessions'):
            sessions = self.get('sessions')
            schemas['sessions'] = get_dataframe_schema(sessions)['items']['properties']
        if table in (None, 'usage'):
            usage = self.get('usage')
            schemas['usage'] = get_dataframe_schema(usage)['items']['properties']
        if table in (None, 'resources'):
            resources = self.get('resources')
            schemas['resources'] = get_dataframe_schema(resources)['items']['properties']
        return schemas if table is None else schemas[table]

    @cached(with_query=False)
    def get(self, table, **query):
        if table not in ('deployments', 'resources', 'sessions', 'usage'):
            raise ValueError(f"AE5Source has no '{table}' table, choose "
                             "from 'deployments', 'resources', 'usage', "
                             "'sessions'.")
        if table == 'deployments':
            deployments = self._session.deployment_list(format='dataframe')
            return deployments[self._deployment_columns]
        elif table == 'resources':
            resources = self._session.resource_profile_list(format='dataframe')
            return resources
        elif table == 'usage':
            records = []
            for deployment in self._session.deployment_list(k8s=True):
                record = self._get_record(deployment)
                if record is None:
                    continue
                records.append(record)
            columns = ['id', 'name', 'url', 'resource_profile', 'cpu',
                       'memory', 'uptime', 'restarts']
            return pd.DataFrame(records, columns=columns)
        else:
            sessions = self._session.session_list(format='dataframe')
            return sessions[self._session_columns]
