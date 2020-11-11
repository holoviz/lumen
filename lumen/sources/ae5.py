import datetime as dt

from concurrent import futures

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
            self.hostname, self.username, self.password, persist=False
        )
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=self.pool_size)
        self._session.session.mount('https://', adapter)
        self._session.session.mount('http://', adapter)

    def get_schema(self, table=None):
        if table in (None, 'deployments'):
            deployments = self.get('deployments')
            deployment_schema = get_dataframe_schema(deployments)['items']['properties']
            if table is not None:
                return deployment_schema
        if table in (None, 'sessions'):
            sessions = self.get('sessions')
            session_schema = get_dataframe_schema(sessions)['items']['properties']
            if table is not None:
                return session_schema
        if table in (None, 'resources'):
            resources = self.get('resources')
            resource_schema = get_dataframe_schema(resources)['items']['properties']
        return {
            'deployments': deployment_schema,
            'resources': resource_schema,
            'sessions': session_schema
        }

    @cached(with_query=False)
    def get(self, table, **query):
        if table not in ('deployments', 'resources', 'sessions'):
            raise ValueError(f"AE5Source has no '{table}' table, choose "
                             "from 'deployments', 'resources' and 'sessions'.")
        if table == 'deployments':
            deployments = self._session.deployment_list(format='dataframe')
            return deployments[self._deployment_columns]
        elif table == 'resources':
            resources = self._session.resource_profile_list(format='dataframe')
            return resources
        else:
            sessions = self._session.session_list(format='dataframe')
            return sessions[self._session_columns]


class AE5KubeSource(Source):
    """
    Queries an endpoint on a Anaconda Enterprise 5 instance for
    information about sessions and deployments.
    """

    ae5_source = param.ClassSelector(class_=AE5Source)

    source_type = 'ae5kube'

    _empty = {
        'containers': [
            {'name': 'app',
             'usage': {'cpu': '', 'mem': ''}
            }
        ]
    }

    def _get_record(self, deployment, resources, pod_info):
        usage_info = pod_info.get('usage', self._empty)
        status_info = pod_info.get('status')

        record = {
            "id": deployment['id'],
            "name": deployment['name'],
            "url": deployment['url'],
        }

        # Resource usage
        containers = [
            container for container in usage_info['containers']
            if container['name'] == 'app'
        ]
        if containers:
            container = containers[0]
            profile = deployment['resource_profile']
            resource = resources[resources['name']==profile]
            usage = container['usage']
            cpu_cap = int(resource.cpu.item())
            if 'cpu' not in usage or 'm' not in usage['cpu']:
                cpu = 0
            else:
                cpu = int(usage['cpu'][:-1])
            record['cpu'] = round(((cpu/1000.) / cpu_cap)*100, 2)
            mem_cap = int(resource.memory.item()[:-2])
            if 'memory' not in usage:
                mem = 0
            else:
                mem = int(usage['memory'][:-2])
            record['memory'] = round(((mem/1024.) / mem_cap)*100, 2)
        else:
            record['cpu'] = float('NaN')
            record['memory'] = float('NaN')

        # Status
        statuses = [
            status for status in status_info['containerStatuses']
            if status['name'] == 'app'
        ]
        if statuses:
            status = statuses[0]
            state = status['state']
            started = state.get('running', state.get('waiting', {})).get('startedAt')
            record["restarts"] = status['restartCount']
            if started:
                started_dt = dt.datetime.fromisoformat(started.replace('Z', ''))
                uptime = str(dt.datetime.now()-started_dt)
                record["uptime"] = uptime[:uptime.index('.')]
        else:
            record["restarts"] = float('NaN')
        if 'uptime' not in record:
            record["uptime"] = '-'

        return record

    def _fetch_data(self, deployments):
        pod_info = {}
        with futures.ThreadPoolExecutor(len(deployments)) as executor:
            tasks = {executor.submit(self.ae5_source._session.pod_info, d['id']): d['id']
                     for i, d in deployments.iterrows()}
            for future in futures.as_completed(tasks):
                deployment = tasks[future]
                try:
                    pod_info[deployment] = future.result()
                except Exception:
                    continue
        return pod_info

    # Public API

    def get_schema(self, table=None):
        deployments = self.get('deployments')
        schema = get_dataframe_schema(deployments)['items']['properties']
        return {'deployments': schema} if table is None else schema

    @cached(with_query=False)
    def get(self, table, **query):
        if table not in ('deployments',):
            raise ValueError(f"AE5KubeSource has no '{table}' table, "
                             "it currently only has a 'deployments' table.")
        deployments = self.ae5_source.get('deployments')
        resources = self.ae5_source.get('resources')
        pod_info = self._fetch_data(deployments)
        records = []
        for i, row in deployments.iterrows():
            record = self._get_record(row, resources, pod_info[row['id']])
            if record is None:
                continue
            records.append(record)
        columns = ['id', 'name', 'url', 'cpu', 'memory', 'uptime', 'restarts']
        return pd.DataFrame(records, columns=columns)
