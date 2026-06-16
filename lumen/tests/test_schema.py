import panel as pn

from panel_material_ui import Checkbox, FloatSlider, Select

from lumen.schema import JSONSchema


def test_boolean_schema():
    json_schema = JSONSchema(schema={'bool': {'type': 'boolean'}}, multi=False)
    assert isinstance(json_schema._widgets['bool'], Checkbox)

def test_number_schema():
    json_schema = JSONSchema(multi=False, schema={
        'number': {'type': 'number', 'inclusiveMinimum': 0, 'inclusiveMaximum': 3.14}
    })
    widget = json_schema._widgets['number']
    assert isinstance(widget, FloatSlider)
    assert widget.start == 0
    assert widget.end == 3.14

def test_enum_schema():
    json_schema = JSONSchema(multi=False, schema={
        'enum': {'type': 'enum', 'enum': ['A', 'B', 'C']}
    })
    widget = json_schema._widgets['enum']
    assert isinstance(widget, Select)
    assert widget.options == ['A', 'B', 'C']
