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

    def __init__(self, **params):
        super().__init__(**params)
        self._session = AEUserSession(
            self.hostname, self.username, self.password, persist=False,
            k8s_endpoint=self.k8s_endpoint
        )
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=self.pool_size)
        self._session.session.mount('https://', adapter)
        self._session.session.mount('http://', adapter)

    def _get_record(self, deployment, resources):
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
        profile = deployment['resource_profile']
        resource = resources[resources['name']==profile]
        usage = container['usage']

        # CPU Usage
        cpu_cap = int(resource.cpu.item())
        if 'cpu' not in usage or 'm' not in usage['cpu']:
            cpu = 0
        else:
            cpu = float(usage['cpu'][:-1])
        record['cpu'] = round(((cpu/1000.) / cpu_cap)*100, 2)

        # Memory usage
        mem_cap = float(resource.memory.item()[:-2])
        mem = float(usage['memory'][:-2]) if 'memory' in usage else 0
        record['memory'] = round(((mem/1024.) / mem_cap)*100, 2)

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
            resources = self.get('resources')
            records = []
            for deployment in self._session.deployment_list(k8s=True):
                record = self._get_record(deployment, resources)
                if record is None:
                    continue
                records.append(record)
            columns = ['id', 'name', 'url', 'cpu', 'memory', 'uptime', 'restarts']
            return pd.DataFrame(records, columns=columns)
        else:
            sessions = self._session.session_list(format='dataframe')
            return sessions[self._session_columns]
