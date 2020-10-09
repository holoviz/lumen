import datetime as dt

import param
import pandas as pd

from ae5_tools.api import AEUserSession

from .base import Source


class AE5KubeSource(Source):
    """
    Queries an endpoint on a Anaconda Enterprise 5 instance for
    information about sessions and deployments.
    """

    url = param.String(doc="URL of the AE5 instance")

    kubernetes_api = param.String(doc="URL of the kube-control deployment")

    username = param.String(doc="Username to authenticate with AE5.")

    password = param.String(doc="Password to authenticate with AE5.")

    source_type = 'ae5'

    _empty = {
        'containers': [
            {'name': 'app',
             'usage': {'cpu': '', 'mem': ''}
            }
        ],
        'timestamp': ''
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._session = AEUserSession(self.url, self.username, self.password, persist=False)
        self._deployment_cache = None
        self._pod_info = {}
        self._deployments = {}
        self._resources = None
        self._update_cache()

    def _get_usage(self, name, d, variable):
        info = self._pod_info[d['id']].get('usage', self._empty)
        containers = [container for container in info['containers'] if container['name'] == 'app']
        if not containers:
            return None
        container = containers[0]
        record = {
            "name": name,
            "state": d['state'],
            "owner": d['owner'],
            "resource_profile": d['resource_profile'],
            "timestamp": info['timestamp']
        }
        resource = self._resources[self._resources['name']==d['resource_profile']]
        usage = container['usage']
        if variable == 'cpu':
            cpu_cap = int(resource.cpu.item())
            if 'cpu' not in usage or 'm' not in usage['cpu']:
                cpu = 0
            else:
                cpu = int(usage['cpu'][:-1])
            record['cpu'] = round(((cpu/1000.) / cpu_cap)*100, 2)
        else:
            mem_cap = int(resource.memory.item()[:-2])
            if 'memory' not in usage:
                mem = 0
            else:
                mem = int(usage['memory'][:-2])
            record['memory'] = round(((mem/1024.) / mem_cap)*100, 2)
        return record

    def _get_uptime(self, name, d, variable):
        info = self._pod_info[d['id']].get('status')
        statuses = [status for status in info['containerStatuses'] if status['name'] == 'app']
        if not statuses:
            return None

        status = statuses[0]
        started = status['state'].get('running', {}).get('startedAt')
        if variable == 'uptime' and started is None:
            return
        record = {
            "name": name,
            "state": d['state'],
            "owner": d['owner'],
            "resource_profile": d['resource_profile'],

        }
        if variable == 'uptime':
            started_dt = dt.datetime.fromisoformat(started.replace('Z', ''))
            uptime = str(dt.datetime.now()-started_dt)
            record['uptime'] = uptime[:uptime.index('.')]
        else:
            record["restarts"] = status['restartCount']
        return record

    def _get_pod_info(self, pod=None):
        url = self.kubernetes_api+'/podinfo'
        if pod is not None:
            url += '?id='+pod
        return self._session.session.get(url).json()

    def _update_cache(self):
        self._deployment_cache = self._session.deployment_list()
        self._resources = self._session.resource_profile_list(format='dataframe')
        self._deployments = {d['name']: d for d in self._deployment_cache}
        usage = dict(self._pod_info)
        for i, d in self._deployments.items():
            usage[d['id']] = self._get_pod_info(d['id'])
        self._pod_info = usage

    # Public API

    def get_schema(self, variable=None):
        name = sorted({d['name'] for d in self._deployment_cache})
        owners = sorted({d['project_owner'] for d in self._deployment_cache})
        state = sorted({d['state'] for d in self._deployment_cache})
        resources = sorted({d['resource_profile'] for d in self._deployment_cache})
        schema = {
            "cpu": {
                "name": {"type": "string", "enum": name},
                "state": {"type": "string", "enum": state},
                "owner": {"type": "string", "enum": owners},
                "resource_profile": {"type": "string", "enum": resources},
                "timestamp": {"type": "string", "format": "datetime"},
                "cpu": {"type": "number"}
            },
            "memory": {
                "name": {"type": "string", "enum": name},
                "state": {"type": "string", "enum": state},
                "owner": {"type": "string", "enum": owners},
                "resource_profile": {"type": "string", "enum": resources},
                "timestamp": {"type": "string", "format": "datetime"},
                "memory": {"type": "number"}
            },
            'uptime': {
                "name": {"type": "string", "enum": name},
                "state": {"type": "string", "enum": state},
                "owner": {"type": "string", "enum": owners},
                "uptime": {"type": "string"},
                "restarts": {"type": "number"}
            },
            'restarts': {
                "name": {"type": "string", "enum": name},
                "state": {"type": "string", "enum": state},
                "owner": {"type": "string", "enum": owners},
                "uptime": {"type": "string"},
                "restarts": {"type": "number"}
            }
        }
        return schema if variable is None else schema[variable]

    def get(self, variable, **query):
        if self._deployment_cache is None:
            self._update_cache()
        records = []
        for name, d in self._deployments.items():
            if variable in ('cpu', 'memory'):
                record = self._get_usage(name, d, variable)
            else:
                record = self._get_uptime(name, d, variable)
            if record is None:
                continue
            skip = False
            for k, v in query.items():
                if record[k] != v:
                    skip = True
                    break
            if skip:
                continue
            records.append(record)
        return pd.DataFrame(records)

    def clear_cache(self):
        self._deployment_cache = None
        self._deployments.clear()
        self._resources = None
        self._pod_info.clear()
