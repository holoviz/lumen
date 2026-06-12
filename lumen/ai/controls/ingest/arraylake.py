from __future__ import annotations

import param

from ....sources.xarray_sql import XArraySQLSource
from ...translate import params_to_callable
from .parametric import ParametricSourceControls
from .result import SourceResult


class ArraylakeSourceControls(ParametricSourceControls):
    """
    Parametric controls that open an Arraylake repository as a
    SQL-queryable xarray source.

    The user supplies a repo, branch, and optional Zarr group; the control
    opens the Icechunk-backed store with xarray and registers the result as
    an ``XArraySQLSource`` (one SQL table per gridded data variable), mirroring
    how uploaded CSVs become a ``DuckDBSource``.

    Example
    -------
    ::

        from lumen.ai.controls.ingest import ArraylakeSourceControls
        ui = ExplorerUI(source_controls=[ArraylakeSourceControls])

    Requires the optional dependencies (Python >=3.12)::

        pip install lumen[arraylake]
    """

    repo = param.String(default="", doc="""
        Arraylake repository, e.g. 'earthmover-public/goes-16'.""")

    branch = param.String(default="main", doc="""
        Branch or ref to read from.""")

    group = param.String(default="", doc="""
        Optional Zarr group within the repository.""")

    variables = param.List(default=None, allow_None=True, precedence=-1, doc="""
        Subset of data variables to expose as tables. When omitted, all
        gridded (non-scalar) variables are exposed.""")

    label = '<span class="material-icons" style="vertical-align: middle;">cloud_queue</span> Arraylake'

    def as_tools(
        self, query: str | None = None, top_k: int = 5,
    ) -> list[tuple[str, callable]]:
        """Expose the repo/branch/group params as a single typed agent tool.

        Mirrors ``URLSourceControls.as_tools``: subclass-pattern controls
        declare class-level params rather than registering actions, so we
        synthesize one callable whose signature mirrors those params for
        ``FunctionTool`` to build a schema from.
        """
        if self._cached_tools is not None:
            return self._cached_tools

        query_names = self._get_query_param_names()
        if not query_names:
            self._cached_tools = []
            return self._cached_tools

        action_name = self.__class__.__name__.removesuffix("Controls")

        async def _tool_callable(**kwargs) -> SourceResult:
            return await self.load_action(action_name, **kwargs)

        query_params = {name: self.param[name] for name in query_names}
        doc = (self.__doc__ or f"Load data from {action_name}.").strip()
        params_to_callable(
            _tool_callable, query_params,
            name=action_name.lower() or "load_data",
            doc=doc,
        )

        self._cached_tools = [(action_name, _tool_callable)]
        return self._cached_tools

    async def _fetch_data(self, action_name: str, **params) -> SourceResult:
        repo = (params.get("repo") or self.repo).strip()
        if not repo:
            return SourceResult.empty(
                "Provide an Arraylake repo, e.g. 'earthmover-public/goes-16'."
            )
        branch = (params.get("branch") or self.branch).strip() or "main"
        group = (params.get("group") or self.group).strip() or None

        try:
            import arraylake as al
            import xarray as xr
        except ImportError:
            return SourceResult.empty(
                "Arraylake support requires `pip install lumen[arraylake]` "
                "(Python >=3.12)."
            )

        # A larger Icechunk snapshot cache avoids repeatedly re-reading metadata
        # nodes while opening large stores; on GOES this cut the open from ~200s
        # to ~16s. Skip cleanly if icechunk is unavailable.
        repo_config = None
        try:
            import icechunk
            repo_config = icechunk.RepositoryConfig(
                caching=icechunk.CachingConfig(num_snapshot_nodes=10_000)
            )
        except Exception:
            pass

        self.progress(f"Opening Arraylake repo {repo!r}")
        try:
            store = al.Client().get_repo(repo, config=repo_config).readonly_session(branch).store
            ds = xr.open_zarr(store, group=group)
        except Exception as e:
            return SourceResult.empty(f"Could not open Arraylake repo {repo!r}: {e}")

        # Drop 0-dim scalar metadata variables: they have no chunkable
        # dimension and cannot become a partitioned table.
        variables = self.variables or [
            name for name, var in ds.data_vars.items() if var.ndim > 0
        ]
        if not variables:
            return SourceResult.empty(
                f"No gridded data variables to load from {repo!r}."
            )

        source_id = f"{self.source_name_prefix}{self._count:06d}"
        self._count += 1
        try:
            # Virtual-zarr stores (e.g. GOES) can chunk variables
            # inconsistently along a shared dimension; unify so the dataset
            # registers as a single coherent set of tables.
            ds = ds.unify_chunks()
            source = XArraySQLSource.from_dataset(
                ds, variables=variables, name=source_id
            )
        except Exception as e:
            return SourceResult.empty(
                f"Could not register {repo!r} as a source: {e}"
            )

        tables = source.get_tables()
        first_table = tables[0] if tables else None
        location = "/".join(part for part in (repo, branch, group) if part)
        message = f"Loaded {len(tables)} tables from Arraylake {location}"
        return SourceResult.from_source(source, table=first_table, message=message)
