import fnmatch
import requests
import datetime as dt

from concurrent import futures
from collections import defaultdict

import param
import pandas as pd
import numpy as np

from ae5_tools.api import AEUserSession

from .base import Source, cached


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

    def _get_record(self, name, d):
        pod_info = self._pod_info[d['id']]
        usage_info = pod_info.get('usage', self._empty)
        status_info = pod_info.get('status')

        record = {
            "name": name,
            "state": d['state'],
            "owner": d['owner'],
            "url": d['url'],
            "resource_profile": d['resource_profile'],
            "timestamp": usage_info['timestamp']
        }

        # Resource usage
        containers = [
            container for container in usage_info['containers']
            if container['name'] == 'app'
        ]
        if containers:
            container = containers[0]
            resource = self._resources[self._resources['name']==d['resource_profile']]
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
        with futures.ThreadPoolExecutor(len(self._deployments)) as executor:
            tasks = {executor.submit(self._get_pod_info, d['id']): d['id']
                     for d in self._deployments.values()}
            for future in futures.as_completed(tasks):
                deployment = tasks[future]
                try:
                    usage[deployment] = future.result()
                except Exception:
                    continue

        self._pod_info = usage

    # Public API

    def get_schema(self, table=None):
        name = sorted({d['name'] for d in self._deployment_cache})
        owners = sorted({d['project_owner'] for d in self._deployment_cache})
        urls = sorted({d['url'] for d in self._deployment_cache})
        state = sorted({d['state'] for d in self._deployment_cache})
        resources = sorted({d['resource_profile'] for d in self._deployment_cache})
        schema = {
            "deployments": {
                "name": {"type": "string", "enum": name},
                "state": {"type": "string", "enum": state},
                "owner": {"type": "string", "enum": owners},
                "resource_profile": {"type": "string", "enum": resources},
                "timestamp": {"type": "string", "format": "datetime"},
                "url": {"type": "string", "enum": urls},
                "cpu": {"type": "number"},
                "memory": {"type": "number"},
                "uptime": {"type": "string"},
                "restarts": {"type": "number"}
            }
        }
        return schema if table is None else schema[table]

    @cached(with_query=False)
    def get(self, table, **query):
        if self._deployment_cache is None:
            self._update_cache()
        records = []
        for name, d in self._deployments.items():
            record = self._get_record(name, d)
            if record is None:
                continue
            records.append(record)
        df = pd.DataFrame(records, columns=[
            'name', 'state', 'owner', 'resource_profile', 'timestamp',
            'url', 'cpu', 'memory', 'uptime', 'restarts'
        ])
        return df

    def clear_cache(self):
        self._cache = {}
        self._deployment_cache = None
        self._deployments.clear()
        self._resources = None
        self._pod_info.clear()


class PrometheusSource(Source):

    hostname = param.String(doc="Hostname of the AE5 instance e.g ae5.organization.com")

    promql_deployment = param.String(default='prometheus', doc="""
       Name of the AE5 deployment exposing the Prometheus API""")

    metrics = param.List(default=['memory_usage', 'cpu_usage', 'network_receive_bytes'],
                         doc="Names of metric queries to execute")

    username = param.String(doc="Username to authenticate with AE5.")

    password = param.String(doc="Password to authenticate with AE5.")

    step = param.String(default='10s', doc="Step value to use in PromQL query_range query")

    deployments = param.List(default=['*'], doc="""
      List of regular expressions over deployment names to match and query""")

    start = param.Parameter(default=dt.timedelta(hours=-1), doc="""
       Integer value (PromQL format), Python datetime for absolute
       timestamp or negative timedelta object that is computed relative
       to end parameter""")

    end = param.Parameter(default=None, allow_None=True, doc="""
       Integer value (PromQL format), Python datetime for absolute
       timestamp or None which represents datetime.now()""")

    source_type = 'prometheus'

    memory_usage_query = """sum by(container_name)
    (container_memory_usage_bytes{job="kubelet",
    cluster="", namespace="default", pod_name=POD_NAME,
    container_name=~"app|app-proxy", container_name!="POD"})"""

    network_receive_bytes_query = """sort_desc(sum by (pod_name)
    (rate(container_network_receive_bytes_total{job="kubelet", cluster="",
    namespace="default", pod_name=POD_NAME}[1m])))"""

    cpu_usage_query = """sum by (container_name)
    (rate(container_cpu_usage_seconds_total{job="kubelet", cluster="",
     namespace="default", image!="", pod_name=POD_NAME,
     container_name=~"app|app-proxy", container_name!="POD"}[1m]))"""

    _metrics = {'memory_usage': {'query':memory_usage_query, 'schema':{"type": "number"}},
                'network_receive_bytes': {'query':network_receive_bytes_query,
                                         'schema':{"type": "number"}},
                'cpu_usage': {'query':cpu_usage_query, 'schema':{"type": "number"}}}

    def __init__(self, **params):
        super().__init__(**params)
        self._session = AEUserSession(self.hostname, self.username, self.password, persist=False)
        self._deployment_cache = None
        self._deployments = {}
        self._update_cache()


    def _format_timestamps(self, start, end):
        '''
        Given a valid start and end value return the formatted
        timestamps to be used in PromQL URL
        '''
        if end is None:
            end = dt.datetime.now()
            end_formatted = end.isoformat("T") + "Z"
        elif isinstance(end, int):
            end_formatted = str(end)
        else:
            end_formatted = end.isoformat("T") + "Z"

        if isinstance(start, dt.timedelta):
            if isinstance(end, int):
                raise Exception('Timedelta can only be used if end is a datetime or None')
            start = end + start

        if isinstance(start, int):
            start_formatted = str(start)
        else:
            start_formatted = start.isoformat("T") + "Z"
        return start_formatted, end_formatted


    def _url_query_parameters(self, pod_id, query, start, end, step):
        "Uses regular expression to map ae5-tools pod_id to full id"
        start_timestamp, end_timestamp = self._format_timestamps(start, end)
        regexp = f'anaconda-app-{pod_id}-.*'
        query = query.replace("pod_name=POD_NAME", f"pod_name=~'{regexp}'")
        query = query.replace("pod=POD_NAME", f"pod=~'{regexp}'")
        query = query.replace('\n',' ')
        query = query.replace(' ','%20')
        query_escaped = query.replace('\"','%22')
        return f'query={query_escaped}&start={start_timestamp}&end={end_timestamp}&step={step}'


    def _get_query_url(self, base_url, metric, pod_id, start, end, step):
        "Return the full query URL"
        query_template = self._metrics[metric]['query']
        query_params = self._url_query_parameters(pod_id, query_template, start, end, step)
        return f'{base_url}/query_range?{query_params}'


    def _get_query_json(self, query_url):
        "Function called in parallel to fetch JSON in ThreadPoolExecutor"
        response = requests.get(query_url, verify=False)
        data = response.json()
        if len(data) == 0:
            return None
        return data


    def _json_to_df(self, metric, response_json):
        "Convert JSON response to pandas DataFrame"
        timestamps = np.array(response_json)[:,0]
        metrics = np.array(response_json)[:,1]
        df = pd.DataFrame({'timestamp':timestamps, metric:metrics})
        df[metric] = df[metric].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.set_index('timestamp')


    def parallel_fetch(self, base_url, metrics, pod_ids, start, end, step):
        "Returns fetched JSON in dictionary indexed by pod_id then metric name"
        triples = [(pod_id, metric,
                    self._get_query_url(base_url, metric, pod_id, start, end, step))
                   for metric in metrics for pod_id in pod_ids]
        fetched_json = defaultdict(dict)
        with futures.ThreadPoolExecutor(len(triples)) as executor:
            tasks = {executor.submit(self._get_query_json, query_url) :
                     (pod_id, metric, query_url)
                     for pod_id, metric, query_url in triples}
            for future in futures.as_completed(tasks):
                (pod_id, metric, query_url) = tasks[future]
                try:
                    fetched_json[pod_id][metric] = future.result()
                except Exception:
                    self.warning(f'Could not fetch {metric} for pod {pod_id}.'
                                 f'Query used: {query_url}')
                    continue

        return fetched_json


    def _make_query(self, **query):
        base_url = f'http://{self.promql_deployment}.{self.hostname}'
        deployments = query.get('deployments', self.deployments)
        step = query.get('step', self.step)
        metrics = query.get('metrics', self.metrics)
        start = query.get('start', self.start)
        end = query.get('end', self.end)

        filtered_names = [k for k in self._deployments.keys()
                          if any(fnmatch.fnmatch(k, d) for d in deployments)]
        if filtered_names == []:
            return None

        pod_ids = [v['id'][3:] for k,v in self._deployments.items() if k in filtered_names]
        # TODO: Remove need for slice
        dfs = []
        json_data = self.parallel_fetch(base_url, metrics, pod_ids, start, end, step)
        for pod_id in pod_ids:
            df = None
            for metric in metrics:
                response_json = json_data[pod_id][metric]
                data_df = self._json_to_df(metric, response_json)
                if df is None:
                    df = data_df
                else:
                    df = pd.merge(df, data_df, on='timestamp')
            url_matches = [v['url'] for v in self._deployment_cache
                           if v['id'].endswith(pod_id)]

            urls = np.full(len(df), url_matches[0])
            df.insert(2, "url", urls, True)
            dfs.append(df.reset_index())
        return pd.concat(dfs)


    def get_schema(self, table=None):
        name = sorted({d['name'] for d in self._deployment_cache})
        owners = sorted({d['project_owner'] for d in self._deployment_cache})
        urls = sorted({d['url'] for d in self._deployment_cache})
        state = sorted({d['state'] for d in self._deployment_cache})
        resources = sorted({d['resource_profile'] for d in self._deployment_cache})
        schema = {"url": {"type": "string"},
                  "timestamp" : {"type": "string", "format": "datetime"}}
        schema = dict({k:mdef['schema'] for k,mdef in self._metrics.items()}, **schema)
        schemas = {"deployments": schema}
        return schemas if table is None else schemas[table]


    @cached(with_query=True)
    def get(self, table=None, **query):
        return self._make_query(**query)


    def _update_cache(self):
        self._deployment_cache = self._session.deployment_list()
        self._deployments = {d['name']: d for d in self._deployment_cache}


    def clear_cache(self):
        super().clear_cache()
        self._deployment_cache = None
        self._deployments = {}