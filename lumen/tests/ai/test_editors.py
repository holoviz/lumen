"""Tests for lumen.ai.editors module."""
import json

from io import BytesIO

import pandas as pd
import param
import pytest

from PIL import Image

import lumen.ai.editors as editors_module

from lumen.ai.editors import LumenEditor, SQLEditor, VegaLiteEditor
from lumen.base import Component


class MockComponent(Component):
    """Minimal component for testing editor exports."""

    data = param.Parameter()

    table = param.String(default="test")


class MockVegaComponent(Component):
    _mock_panel = param.Parameter()

    def get_panel(self):
        return self._mock_panel


def _make_png_bytes(mode="RGBA"):
    img = Image.new(mode, (10, 10), color=(255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_MINIMAL_VEGALITE_SPEC = """\
$schema: https://vega.github.io/schema/vega-lite/v5.json
mark: bar
encoding:
  x:
    field: A
    type: quantitative
  y:
    field: B
    type: quantitative
data:
  values:
    - {A: 1, B: 2}
"""


@pytest.fixture
def mock_panel():
    class FakePanel:
        def __init__(self):
            self.calls = []

        def export(self, fmt, **kwargs):
            self.calls.append((fmt, kwargs))
            return _make_png_bytes()

    return FakePanel()


@pytest.fixture
def vegalite_editor(monkeypatch, mock_panel):
    monkeypatch.setattr(editors_module, 'ParamMethod', lambda *args, **kwargs: None)
    monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)
    component = MockVegaComponent(_mock_panel=mock_panel)
    return VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)


@pytest.fixture
def sql_editor(monkeypatch):
    """Return a SQLEditor with a minimal valid component and spec."""
    df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 25]})
    component = MockComponent(data=df)
    monkeypatch.setattr(editors_module, 'ParamMethod', lambda *args, **kwargs: None)
    return SQLEditor(component=component, spec="SELECT * FROM test")


def test_sqleditor_export_sql(sql_editor):
    result = sql_editor.export('sql')
    assert 'SELECT' in result.getvalue()


def test_sqleditor_export_csv(sql_editor):
    result = sql_editor.export('csv')
    content = result.getvalue()
    assert 'name' in content
    assert 'Alice' in content


def test_sqleditor_export_json(sql_editor):
    result = sql_editor.export('json')
    data = json.loads(result.getvalue())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]['name'] == 'Alice'


def test_sqleditor_export_markdown(sql_editor):
    result = sql_editor.export('markdown')
    content = result.getvalue()
    assert '|' in content  # markdown tables use pipes
    assert 'name' in content
    assert 'Alice' in content



def test_vegalite_export_formats_includes_pillow_formats():
    assert {"webp", "tiff", "eps"} <= set(VegaLiteEditor.export_formats)


@pytest.mark.parametrize("fmt,validate", [
    ("webp", lambda r: Image.open(r).format == "WEBP"),
    ("tiff", lambda r: Image.open(r).format == "TIFF"),
    ("eps",  lambda r: r.read().startswith(b"%!PS")),
])
def test_vegalite_pillow_export(vegalite_editor, mock_panel, fmt, validate):
    result = vegalite_editor.export(fmt)
    assert isinstance(result, BytesIO)
    assert validate(result)
    assert mock_panel.calls[-1] == ("png", {"scale": 2})


def test_class_name_to_filename():
    """Test CamelCase to snake_case conversion for base class."""
    assert LumenEditor._class_name_to_download_filename("yaml") == "lumen_editor.yaml"
    assert LumenEditor._class_name_to_download_filename("json") == "lumen_editor.json"


def test_class_name_with_acronyms():
    """Test handling of acronyms in class names."""
    class SQLEditor(LumenEditor):
        pass
    
    assert SQLEditor._class_name_to_download_filename("yaml") == "sql_editor.yaml"


def test_class_name_with_numbers():
    """Test handling of numbers in class names."""
    class Editor2D(LumenEditor):
        pass

    assert Editor2D._class_name_to_download_filename("png") == "editor2_d.png"


# --- Tests for zoom/pan bounds capture in export (issue #1773) ---

def _make_mock_selection(**fields):
    """Create a mock selection object with interval selection params."""
    params = {
        name: param.Dict(default=None) for name in fields
    }
    sel = type('Selection', (param.Parameterized,), params)()
    for name, value in fields.items():
        setattr(sel, name, value)
    return sel


class TestGetCurrentBounds:

    def test_returns_bounds_from_live_panel(self, monkeypatch):
        monkeypatch.setattr(editors_module, 'ParamMethod', lambda *a, **kw: None)
        monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)

        mock_panel = type('MockPanel', (), {'selection': None})()
        mock_panel.selection = _make_mock_selection(
            brush={'A': [2, 8], 'B': [10, 50]}
        )

        component = MockVegaComponent()
        component._panel = mock_panel

        editor = VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)
        editor.component = component  # restore after __init__ may have replaced it

        assert editor._get_current_bounds() == {'A': [2, 8], 'B': [10, 50]}

    def test_returns_none_when_no_panel(self, monkeypatch):
        monkeypatch.setattr(editors_module, 'ParamMethod', lambda *a, **kw: None)
        monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)

        component = MockVegaComponent()
        component._panel = None

        editor = VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)
        editor.component = component

        assert editor._get_current_bounds() is None

    def test_returns_none_when_no_selection(self, monkeypatch):
        monkeypatch.setattr(editors_module, 'ParamMethod', lambda *a, **kw: None)
        monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)

        mock_panel = type('MockPanel', (), {'selection': None})()

        component = MockVegaComponent()
        component._panel = mock_panel

        editor = VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)
        editor.component = component

        assert editor._get_current_bounds() is None

    def test_ignores_non_dict_selections(self, monkeypatch):
        """Point selections (lists) should be ignored, only interval (dict) kept."""
        monkeypatch.setattr(editors_module, 'ParamMethod', lambda *a, **kw: None)
        monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)

        params = {
            'zoom': param.Dict(default=None),
            'click': param.List(default=None),
        }
        sel = type('Selection', (param.Parameterized,), params)()
        sel.zoom = {'A': [1, 5]}
        sel.click = [1, 2, 3]

        mock_panel = type('MockPanel', (), {'selection': sel})()

        component = MockVegaComponent()
        component._panel = mock_panel

        editor = VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)
        editor.component = component

        assert editor._get_current_bounds() == {'A': [1, 5]}


class TestSetSelectionValue:
    """Tests for the primary strategy: injecting bounds via selection param value."""

    def test_sets_value_on_interval_selection_with_bind_scales(self):
        spec = {
            'params': [
                {'name': 'zoom', 'select': {'type': 'interval'}, 'bind': 'scales'}
            ],
            'encoding': {'x': {'field': 'A'}, 'y': {'field': 'B'}},
        }
        bounds = {'A': [2, 8], 'B': [10, 50]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['params'][0]['value'] == {'A': [2, 8], 'B': [10, 50]}
        # Should NOT also inject scale.domain (selection value is sufficient)
        assert 'scale' not in result['encoding']['x']

    def test_geographic_map_no_selection_no_mutation(self):
        """Geographic maps don't use bind='scales' (Vega-Lite doesn't support it
        with projections). Bounds injection is a no-op; the fix for geographic
        maps is dimension preservation in export()."""
        spec = {
            'projection': {'type': 'equalEarth'},
            'layer': [
                {'mark': 'geoshape'},
                {
                    'mark': 'circle',
                    'encoding': {
                        'longitude': {'field': 'Longitude', 'type': 'quantitative'},
                        'latitude': {'field': 'Latitude', 'type': 'quantitative'},
                    }
                }
            ]
        }
        bounds = {'Longitude': [-130, -60], 'Latitude': [10, 55]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        # No selection param to set, no x/y encoding to inject into
        # Spec should pass through unchanged (geographic fix is via dimensions)
        assert result['projection'] == {'type': 'equalEarth'}
        assert 'params' not in result
        assert 'scale' not in result['layer'][1]['encoding'].get('longitude', {})

    def test_ignores_non_interval_selections(self):
        spec = {
            'params': [
                {'name': 'click', 'select': {'type': 'point'}},
            ],
            'encoding': {'x': {'field': 'A', 'type': 'quantitative'}},
        }
        bounds = {'A': [2, 8]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        # Should NOT set value on point selection
        assert 'value' not in result['params'][0]
        # Should fall back to scale.domain
        assert result['encoding']['x']['scale']['domain'] == [2, 8]

    def test_ignores_interval_without_bind_scales(self):
        spec = {
            'params': [
                {'name': 'brush', 'select': {'type': 'interval'}},
            ],
            'encoding': {'x': {'field': 'A', 'type': 'quantitative'}},
        }
        bounds = {'A': [2, 8]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        # No bind="scales" → not a zoom selection, fall back
        assert 'value' not in result['params'][0]
        assert result['encoding']['x']['scale']['domain'] == [2, 8]

    def test_selection_in_layer(self):
        """Selection defined inside a layer, not at top level."""
        spec = {
            'layer': [
                {
                    'params': [
                        {'name': 'zoom', 'select': {'type': 'interval'}, 'bind': 'scales'}
                    ],
                    'mark': 'point',
                    'encoding': {'x': {'field': 'A'}, 'y': {'field': 'B'}},
                },
                {'mark': 'line', 'encoding': {'x': {'field': 'A'}, 'y': {'field': 'C'}}},
            ]
        }
        bounds = {'A': [1, 5], 'B': [10, 30]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['layer'][0]['params'][0]['value'] == {'A': [1, 5], 'B': [10, 30]}

    def test_selection_string_type(self):
        """Vega-Lite also accepts select as a string shorthand."""
        spec = {
            'params': [
                {'name': 'zoom', 'select': 'interval', 'bind': 'scales'}
            ],
        }
        bounds = {'A': [2, 8]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['params'][0]['value'] == {'A': [2, 8]}

    def test_does_not_mutate_original(self):
        spec = {
            'params': [
                {'name': 'zoom', 'select': {'type': 'interval'}, 'bind': 'scales'}
            ],
        }
        bounds = {'A': [2, 8]}
        VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert 'value' not in spec['params'][0]

    def test_empty_bounds_no_change(self):
        spec = {
            'params': [
                {'name': 'zoom', 'select': {'type': 'interval'}, 'bind': 'scales'}
            ],
        }
        result = VegaLiteEditor._apply_bounds_to_spec(spec, {})

        assert 'value' not in result['params'][0]


class TestEncodingDomainFallback:
    """Tests for the fallback strategy: scale.domain injection on encodings."""

    def test_simple_spec_without_selection(self):
        spec = {
            'encoding': {
                'x': {'field': 'A', 'type': 'quantitative'},
                'y': {'field': 'B', 'type': 'quantitative'},
            }
        }
        bounds = {'A': [2, 8], 'B': [10, 50]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['encoding']['x']['scale']['domain'] == [2, 8]
        assert result['encoding']['y']['scale']['domain'] == [10, 50]

    def test_preserves_existing_scale_properties(self):
        spec = {
            'encoding': {
                'x': {'field': 'A', 'type': 'quantitative', 'scale': {'type': 'log'}},
            }
        }
        bounds = {'A': [1, 100]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['encoding']['x']['scale']['domain'] == [1, 100]
        assert result['encoding']['x']['scale']['type'] == 'log'

    def test_layered_spec_without_selection(self):
        spec = {
            'encoding': {'x': {'field': 'A', 'type': 'quantitative'}},
            'layer': [
                {'encoding': {'y': {'field': 'B', 'type': 'quantitative'}}, 'mark': 'line'},
                {'encoding': {'y': {'field': 'C', 'type': 'quantitative'}}, 'mark': 'point'},
            ]
        }
        bounds = {'A': [1, 5], 'B': [10, 20]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['encoding']['x']['scale']['domain'] == [1, 5]
        assert result['layer'][0]['encoding']['y']['scale']['domain'] == [10, 20]
        assert 'scale' not in result['layer'][1]['encoding']['y']

    def test_no_matching_fields(self):
        spec = {
            'encoding': {
                'x': {'field': 'A', 'type': 'quantitative'},
                'y': {'field': 'B', 'type': 'quantitative'},
            }
        }
        bounds = {'Z': [0, 100]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert 'scale' not in result['encoding']['x']
        assert 'scale' not in result['encoding']['y']

    def test_concat_spec(self):
        spec = {
            'hconcat': [
                {'encoding': {'x': {'field': 'A', 'type': 'quantitative'}}, 'mark': 'bar'},
                {'encoding': {'x': {'field': 'B', 'type': 'quantitative'}}, 'mark': 'bar'},
            ]
        }
        bounds = {'A': [0, 10]}
        result = VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert result['hconcat'][0]['encoding']['x']['scale']['domain'] == [0, 10]
        assert 'scale' not in result['hconcat'][1]['encoding']['x']

    def test_does_not_mutate_original(self):
        spec = {
            'encoding': {
                'x': {'field': 'A', 'type': 'quantitative'},
            }
        }
        bounds = {'A': [2, 8]}
        VegaLiteEditor._apply_bounds_to_spec(spec, bounds)

        assert 'scale' not in spec['encoding']['x']


class TestExportWithBounds:

    def test_export_injects_bounds(self, monkeypatch, mock_panel):
        """Verify export() reads bounds and produces output without error."""
        monkeypatch.setattr(editors_module, 'ParamMethod', lambda *a, **kw: None)
        monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)

        component = MockVegaComponent(_mock_panel=mock_panel)
        editor = VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)

        # Simulate zoom state on the live panel
        sel = _make_mock_selection(zoom={'A': [2, 5], 'B': [10, 30]})
        live_panel = type('LivePanel', (), {'selection': sel, 'width': None, 'height': None})()
        editor.component._panel = live_panel

        result = editor.export('png')
        assert isinstance(result, BytesIO)
        assert len(mock_panel.calls) == 1

    def test_export_without_bounds_unchanged(self, vegalite_editor, mock_panel):
        """Verify export works normally when no zoom state exists."""
        result = vegalite_editor.export('png')
        assert isinstance(result, BytesIO)
        assert len(mock_panel.calls) == 1

    def test_export_uses_live_panel_dimensions(self, monkeypatch, mock_panel):
        """Verify export uses live panel width/height when available."""
        monkeypatch.setattr(editors_module, 'ParamMethod', lambda *a, **kw: None)
        monkeypatch.setattr(VegaLiteEditor, '_update_component', lambda self, *a, **kw: None)

        component = MockVegaComponent(_mock_panel=mock_panel)
        editor = VegaLiteEditor(component=component, spec=_MINIMAL_VEGALITE_SPEC)

        live_panel = type('LivePanel', (), {
            'selection': None, 'width': 1200, 'height': 600
        })()
        editor.component._panel = live_panel

        result = editor.export('png')
        assert isinstance(result, BytesIO)
        assert len(mock_panel.calls) == 1
