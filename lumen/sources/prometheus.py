import datetime as dt
import urllib.parse as urlparse

from collections import defaultdict
from concurrent import futures

import pandas as pd
import panel as pn
import param
import requests

from .base import Source, cached
from ..util import parse_timedelta


class PrometheusSource(Source):
    """
    Queries a Prometheus PromQL endpoint for timeseries information
    about Kubernetes pods.
    """

    ae5_source = param.Parameter(doc="""
      An AE5Source instance to use for querying.""")

    ids = param.List(default=[], doc="""
      List of pod IDs to query.""")

    metrics = param.List(
        default=['memory_usage', 'cpu_usage', 'restarts',
                 'network_receive_bytes', 'network_transmit_bytes'],
        doc="Names of metric queries to execute")

    promql_api = param.String(doc="""
      Name of the AE5 deployment exposing the Prometheus API""")

    period = param.String(default='3h', doc="""
      Period to query over specified as a string. Supports:

        - Week:   '1w'
        - Day:    '1d'
        - Hour:   '1h'
        - Minute: '1m'
        - Second: '1s'
    """)

    samples = param.Integer(default=200, doc="""
      Number of samples in the selected period to query. May be
      overridden by explicit step value.""")

    shared = param.Boolean(default=False, readonly=True, doc="""
      PrometheusSource cannot be shared because it has per-user state.""")

    step = param.String(doc="""
        Step value to use in PromQL query_range query.""")

    source_type = 'prometheus'

    _memory_usage_query = """sum by(container_name)
    (container_memory_usage_bytes{job="kubelet",
    cluster="", namespace="default", pod_name=POD_NAME,
    container_name=~"app", container_name!="POD"})"""

    _network_receive_bytes_query = """sort_desc(sum by (pod_name)
    (rate(container_network_receive_bytes_total{job="kubelet", cluster="",
    namespace="default", pod_name=POD_NAME}[1m])))"""

    _network_transmit_bytes_query = """sort_desc(sum by (pod_name)
    (rate(container_network_transmit_bytes_total{job="kubelet", cluster="",
    namespace="default", pod_name=POD_NAME}[1m])))"""

    _cpu_usage_query = """sum by (container_name)
    (rate(container_cpu_usage_seconds_total{job="kubelet", cluster="",
     namespace="default", image!="", pod_name=POD_NAME,
     container_name=~"app", container_name!="POD"}[1m]))"""

    _restarts_query = """max by (container)
     (kube_pod_container_status_restarts_total{job="kube-state-metrics",
    cluster="", namespace="default", pod=POD_NAME,
    container=~"app"})"""

    _metrics = {
        'memory_usage': {
            'query': _memory_usage_query,
            'schema': {"type": "number"}
        },
        'network_receive_bytes': {
            'query': _network_receive_bytes_query,
            'schema': {"type": "number"}
        },
        'network_transmit_bytes': {
            'query': _network_transmit_bytes_query,
            'schema': {"type": "number"}
        },
        'cpu_usage': {
            'query': _cpu_usage_query,
            'schema': {"type": "number"}
        },
        'restarts': {
            'query': _restarts_query,
            'schema': {"type": "number"}
        }
    }

    @property
    def panel(self):
        return pn.Param(self.param, parameters=['period', 'samples', 'step'],
                        sizing_mode='stretch_width', show_name=False)

    def _format_timestamps(self):
        end = dt.datetime.utcnow()
        end_formatted = end.isoformat("T") + "Z"
        period = parse_timedelta(self.period)
        if period is None:
            raise ValueError(f"Could not parse period '{self.period}'. "
                             "Must specify weeks ('1w'), days ('1d'), "
                             "hours ('1h'), minutes ('1m'), or "
                             "seconds ('1s').")
        start = end - period
        start_formatted = start.isoformat("T") + "Z"
        return start_formatted, end_formatted

    def _step_value(self):
        if self.step:
            return self.step
        return int((parse_timedelta(self.period)/self.samples).total_seconds())

    def _url_query_parameters(self, pod_id, query):
        """
        Uses regular expression to map ae5-tools pod_id to full id.
        """
        start_timestamp, end_timestamp = self._format_timestamps()
        regexp = f'anaconda-app-{pod_id}-.*'
        query = query.replace("pod_name=POD_NAME", f"pod_name=~'{regexp}'")
        query = query.replace("pod=POD_NAME", f"pod=~'{regexp}'")
        query_params = {
            'query': query, 'start': start_timestamp,
            'end': end_timestamp, 'step': self._step_value()
        }
        return urlparse.urlencode(query_params)

    def _get_query_url(self, metric, pod_id):
        "Return the full query URL"
        query_template = self._metrics[metric]['query']
        query_params = self._url_query_parameters(pod_id, query_template)
        if self.ae5_source:
            ae5 = self.ae5_source._session
            return f'https://{ae5._k8s_endpoint}.{ae5.hostname}/promql/query_range?{query_params}'
        else:
            return f'{self.promql_api}/query_range?{query_params}'

    def _get_query_json(self, query_url):
        "Function called in parallel to fetch JSON in ThreadPoolExecutor"
        if self.ae5_source:
            response = self.ae5_source._session.session.get(query_url, verify=False)
        else:
            response = requests.get(query_url, verify=False)
        data = response.json()
        if len(data) == 0:
            return None
        return data

    def _json_to_df(self, metric, values):
        "Convert JSON response to pandas DataFrame"
        df = pd.DataFrame(values, columns=['timestamp', metric])
        df[metric] = df[metric].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.set_index('timestamp')

    def _fetch_data(self, pod_ids):
        "Returns fetched JSON in dictionary indexed by pod_id then metric name"
        if not pod_ids:
            return {}
        # ToDo: Remove need to slice
        triples = [
            (pod_id, metric, self._get_query_url(metric, pod_id[3:]))
            for pod_id in pod_ids for metric in self.metrics
        ]
        fetched_json = defaultdict(dict)
        with futures.ThreadPoolExecutor(len(triples)) as executor:
            tasks = {
                executor.submit(self._get_query_json, t[2]): t
                for t in triples
            }
            for future in futures.as_completed(tasks):
                (pod_id, metric, query_url) = tasks[future]
                try:
                    fetched_json[pod_id][metric] = future.result()
                except Exception as e:
                    fetched_json[pod_id][metric] = []
                    self.param.warning(
                        f"Could not fetch {metric} for pod {pod_id}. "
                        f"Query used: {query_url}, errored with {type(e)}({e})."
                    )
        return fetched_json

    def _make_query(self):
        json_data = self._fetch_data(self.ids)
        dfs = []
        for pod_id, pod_data in json_data.items():
            df = None
            for metric in self.metrics:
                data_df = self._json_to_df(metric, pod_data[metric])
                if df is None:
                    df = data_df
                else:
                    df = pd.merge(df, data_df, on='timestamp', how='outer')
            df = df.reset_index()
            df.insert(0, 'id', pod_id)
            dfs.append(df)
        if dfs:
            return pd.concat(dfs)
        else:
            return pd.DataFrame(columns=list(self.get_schema('timeseries')))

    def get_schema(self, table=None):
        dt_start, dt_end = self._format_timestamps()
        schema = {
            "id": {
                "type": "string",
                "enum": self.ids
            },
            "timestamp" : {
                "type": "string",
                "format": "date-time",
                "formatMinimum": dt_start,
                "formatMaximum": dt_end
            }
        }
        for k, mdef in self._metrics.items():
            schema[k] = mdef['schema']
        return {"timeseries": schema} if table is None else schema

    @cached(with_query=False)
    def get(self, table, **query):
        if table not in ('timeseries',):
            raise ValueError(f"PrometheusSource has no '{table}' table, "
                             "it currently only has a 'timeseries' table.")
        return self._make_query()
