import datetime as dt

import param
import requests

from ae5_tools.api import AEAdminSession, AEUserSession
from panel import state

from .base import Source, cached
from ..util import get_dataframe_schema


class AE5Source(Source):
    """
    The AE5Source provides a number of tables, which allow monitoring
    the nodes, deployments, sessions and resource profiles of an
    Anaconda Enterprise 5 installation.
    """

    hostname = param.String(doc="URL of the AE5 host.")

    username = param.String(doc="Username to authenticate with AE5.")

    password = param.String(doc="Password to authenticate with AE5.")

    admin_username = param.String(doc="Username to authenticate admin with AE5.")

    admin_password = param.String(doc="Password to authenticate admin with AE5.")

    k8s_endpoint = param.String(default='k8s')

    pool_size = param.Integer(default=100, doc="""
      Size of HTTP socket pool.""")

    private = param.Boolean(default=True, doc="""
      Whether to limit the deployments visible to a user based on
      their authorization.""")

    source_type = 'ae5'

    _deployment_columns = [
        'id', 'name', 'url', 'owner', 'resource_profile', 'public', 'state',
        'cpu', 'cpu_percent', 'memory', 'memory_percent', 'uptime', 'restarts'
    ]

    _session_columns = [
        'id', 'name', 'url', 'owner', 'resource_profile', 'state'
    ]

    _tables = ['deployments', 'nodes', 'resources', 'sessions']

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
        if self.admin_username:
            self._admin_session = AEAdminSession(
                self.hostname, self.admin_username, self.admin_password
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

    @property
    def _user(self):
        return state.headers.get('Anaconda-User') if self.private else None

    def _process_deployment(self, deployment):
        container_info = deployment.get('_k8s', {}).get('containers', {})
        if 'app' not in container_info:
            nan = float('NaN')
            deployment['cpu'] = nan
            deployment['memory'] = nan
            deployment['restarts'] = nan
            deployment['uptime'] = '-'
            return deployment

        container = container_info['app']
        limits = container['limits']
        usage = container['usage']

        # CPU Usage
        cpu_cap = self._convert_value(limits['cpu'])
        if 'cpu' in usage:
            cpu = self._convert_value(usage['cpu'])
            deployment['cpu'] = cpu
            deployment['cpu_percent'] = round((cpu / cpu_cap)*100, 2)
        else:
            deployment['cpu_percent'] = float('nan')
            deployment['cpu'] = float('nan')

        # Memory usage
        mem_cap = self._convert_value(limits['memory'])
        if 'memory' in usage:
            mem = self._convert_value(usage['memory'])
            deployment['memory'] = mem
            deployment['memory_percent'] = round((mem / mem_cap)*100, 2)
        else:
            deployment['memory'] = float('nan')
            deployment['memory_percent'] = float('nan')

        # Uptime
        started = container.get('since')
        deployment["restarts"] = container['restarts']
        if started:
            started_dt = dt.datetime.fromisoformat(started.replace('Z', ''))
            uptime = str(dt.datetime.now()-started_dt)
            deployment["uptime"] = uptime[:uptime.index('.')]
        else:
            deployment["uptime"] = '-'

        return deployment

    def _process_nodes(self, node):
        caps = {
            'cpu': self._convert_value(node['capacity/cpu']),
            'gpu': self._convert_value(node['capacity/gpu']),
            'mem': self._convert_value(node['capacity/mem']),
            'pod': int(node['capacity/pod'])
        }
        for column in node.index:
            if 'capacity' in column or '/' not in column:
                continue
            vtype = column.split('/')[1]
            cap = caps[vtype]
            value = node[column]
            if vtype != 'pod':
                value = self._convert_value(value)
            node[column] = value
            node[f'{column}_percent'] = 0 if cap == 0 else round((value/caps[vtype])*100, 2)
        return node

    def _get_deployments(self):
        user = self._user
        deployments = self._session.deployment_list(
            k8s=True, format='dataframe', collaborators=bool(user)
        ).apply(self._process_deployment, axis=1)
        if self.admin_username and user is not None:
            self._admin_session.authorize()
            user_id = self._admin_session.user_info(user)['id']
            roles = self._admin_session._get(f'users/{user_id}/role-mappings/realm/composite')
            is_admin = any(role['name'] == 'ae-admin' for role in roles)
            groups = [g['name'] for g in self._admin_session._get(f'users/{user_id}/groups')]
        else:
            is_admin = False
            groups = []
        if user is None or is_admin:
            return deployments[self._deployment_columns]
        return deployments[
            deployments.public |
            (deployments.owner == user) |
            deployments._collaborators.apply(
                lambda cs: any(c['id'] in groups if c['type'] == 'group' else c['id'] == user for c in cs)
            )
        ][self._deployment_columns]

    def _get_nodes(self):
        nodes = self._session.node_list(format='dataframe').apply(self._process_nodes, axis=1)
        return nodes[[c for c in nodes.columns if not c.startswith('_')]]

    def _get_resources(self):
        return self._session.resource_profile_list(format='dataframe')

    def _get_sessions(self):
        sessions = self._session.session_list(format='dataframe')
        if self._user:
            sessions = sessions[sessions.owner==self._user]
        return sessions[self._session_columns]

    @cached(with_query=False)
    def get(self, table, **query):
        if table not in self._tables:
            raise ValueError(f"AE5Source has no '{table}' table, choose from {repr(self._tables)}.")
        return getattr(self, f'_get_{table}')()

    def get_schema(self, table=None):
        schemas = {}
        for t in self._tables:
            if table is None or t == table:
                schemas[t] = get_dataframe_schema(self.get(t))['items']['properties']
        return schemas if table is None else schemas[table]

    def get_tables(self):
        return self._tables
