"""
The core install (sources, pipelines, views, YAML dashboards) must work
without the dependencies of the ai extra. Each test runs in a subprocess
that blocks those modules from being imported.
"""
import pathlib
import re
import subprocess
import sys
import textwrap
import tomllib

import pytest

from lumen.util import AI_EXTRA_MODULES

ROOT = pathlib.Path(__file__).parents[2]
CSV = ROOT / 'lumen' / 'tests' / 'sources' / 'test.csv'

BLOCKER = textwrap.dedent(f"""
    import importlib.abc
    import importlib.util
    import sys

    BLOCKED = {list(AI_EXTRA_MODULES)!r}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.partition('.')[0] in BLOCKED:
                raise ModuleNotFoundError(f'No module named {{name!r}}', name=name)

    sys.meta_path.insert(0, _Blocker())
""")

CHECK_BOUNDARY = textwrap.dedent("""
    leaked = sorted(m for m in sys.modules if m.partition('.')[0] in BLOCKED)
    assert not leaked, leaked
    loaded_ai = sorted(m for m in sys.modules if m == 'lumen.ai' or m.startswith('lumen.ai.'))
    assert not loaded_ai, loaded_ai
""")


def run_without_ai(code, tmp_path):
    script = tmp_path / 'script.py'
    script.write_text(BLOCKER + textwrap.dedent(code) + CHECK_BOUNDARY)
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        cwd=tmp_path, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _names(deps):
    return {re.match(r'[\w.-]+', dep).group().lower() for dep in deps}


def test_ai_extra_matches_blocked_modules():
    path = ROOT / 'pyproject.toml'
    if not path.exists():
        pytest.skip('pyproject.toml is only available in a source checkout')
    project = tomllib.loads(path.read_text())['project']
    core = _names(project['dependencies'])
    ai = _names(project['optional-dependencies']['ai'])
    assert not core & ai
    distributions = {
        'markdown_it': 'markdown-it-py', 'panel_splitjs': 'panel-splitjs',
        'pydantic_extra_types': 'pydantic-extra-types', 'vl_convert': 'vl-convert-python',
    }
    assert {distributions.get(m, m) for m in AI_EXTRA_MODULES} == ai


def test_import_core_without_ai(tmp_path):
    run_without_ai("""
        import lumen
        import lumen.command
        import lumen.filters
        import lumen.sources
        import lumen.transforms
        import lumen.views

        if importlib.util.find_spec('duckdb'):
            import lumen.sources.duckdb
    """, tmp_path)


def test_pipeline_without_ai(tmp_path):
    run_without_ai(f"""
        from lumen.filters.base import ConstantFilter
        from lumen.pipeline import Pipeline
        from lumen.sources.base import FileSource
        from lumen.views.base import Table

        source = FileSource(tables={{'test': {str(CSV)!r}}})
        pipeline = Pipeline(source=source, table='test')
        pipeline.add_filter(ConstantFilter(field='A', value=(0, 2)))
        assert len(pipeline.data) == 3
        Table(pipeline=pipeline).get_panel()
    """, tmp_path)


def test_serve_yaml_without_ai(tmp_path):
    spec = tmp_path / 'dashboard.yaml'
    spec.write_text(textwrap.dedent(f"""
        sources:
          test:
            type: file
            tables:
              test: {CSV}
        pipelines:
          test:
            source: test
            table: test
            filters:
              - type: widget
                field: D
        layouts:
          - title: Test
            pipeline: test
            views:
              - type: table
              - type: hvplot
                kind: scatter
                x: A
                y: B
    """))
    run_without_ai(f"""
        from bokeh.document import Document
        from lumen.command import YamlHandler

        from lumen.dashboard import Dashboard

        handler = YamlHandler(filename={str(spec)!r})
        handler.modify_document(Document())
        assert not handler.failed, handler.error_detail

        dashboard = Dashboard({str(spec)!r})
        dashboard._render_dashboard()
        views = dashboard.layouts[0]._cards[0].views
        assert [type(view).__name__ for view in views] == ['Table', 'hvPlotView']
        views[1].get_panel()
    """, tmp_path)


def test_cli_without_ai(tmp_path):
    run_without_ai("""
        import lumen.command

        calls = []
        lumen.command.main = lambda args=None: calls.append(sys.argv[1:])
        for argv in (['serve', 'a.yaml', '--port', '0'], ['serve', 'a.yml', 'b.yaml'], ['--version']):
            lumen.command.ai_main(argv)
            assert calls.pop() == argv

        for argv in (['serve', 'data.csv'], ['serve', 'a.yaml', 'data.csv'], []):
            try:
                lumen.command.ai_main(argv)
            except SystemExit as e:
                assert e.code == 1
            else:
                raise AssertionError(f'expected SystemExit for {argv}')
        assert not calls
    """, tmp_path)


def test_import_lumen_ai_without_ai_hints_at_extra(tmp_path):
    result = subprocess.run(
        [sys.executable, '-c', BLOCKER + textwrap.dedent("""
            try:
                import lumen.ai
            except ModuleNotFoundError as e:
                assert "pip install 'lumen[ai]'" in str(e), e
                assert e.name in BLOCKED, e.name
            else:
                raise AssertionError('lumen.ai imported')
        """)], capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_read_geojson_without_ai(tmp_path):
    pytest.importorskip('geopandas')
    geojson = tmp_path / 'points.geojson'
    geojson.write_text(
        '{"type": "FeatureCollection", "features": ['
        '{"type": "Feature", "properties": {"name": "a"}, "geometry": {"type": "Point", "coordinates": [1, 2]}},'
        '{"type": "Feature", "properties": {"name": "b"}, "geometry": {"type": "Point", "coordinates": [3, 4]}}]}'
    )
    run_without_ai(f"""
        import lumen

        source = lumen.read({str(geojson)!r})
        assert source.get_tables() == ['points']
        assert len(source.get('points')) == 2
    """, tmp_path)
