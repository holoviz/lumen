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

    def _get_record(self, deployment):
        container_info = deployment.get('_k8s', {}).get('containers', {})
        record = {
            "id": deployment['id'],
            "name": deployment['name'],
            "url": deployment['url'],
        }

        if 'app' not in container_info:
            nan = float('NaN')
            return dict(record, cpu=nan, memory=nan, restarts=nan, uptime='-')

        container = container_info['app']
        limits = container['limits']
        usage = container['usage']

        # CPU Usage
        cpu_cap_str = limits['cpu']
        if cpu_cap_str.endswith('m'):
            cpu_cap = float(cpu_cap_str[:-1])/1000
        else:
            cpu_cap = float(cpu_cap_str)
        if 'cpu' in usage:
            cpu_str = usage['cpu']
            if cpu_str.endswith('m'):
                cpu = float(cpu_str[:-1])/1000
            else:
                cpu = float(cpu_str)
        else:
            cpu = float('nan')
        record['cpu'] = round((cpu / cpu_cap)*100, 2)

        # Memory usage
        mem_cap_str = limits['memory']
        mem_unit = [m for m in self._units if m is not None and mem_cap_str.endswith(m)]
        mem_unit = mem_unit[0] if mem_unit else None
        mem_cap = float(mem_cap_str[:-len(mem_unit)])*self._units.get(mem_unit, 1)
        if 'memory' in usage:
            mem_str = usage['memory']
            mem = float(mem_str[:-2])*self._units[mem_str[-2:]]
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
            columns = ['id', 'name', 'url', 'cpu', 'memory', 'uptime', 'restarts']
            return pd.DataFrame(records, columns=columns)
        else:
            sessions = self._session.session_list(format='dataframe')
            return sessions[self._session_columns]
