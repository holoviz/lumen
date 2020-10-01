import panel as pn

from lumen.schema import JSONSchema


def test_boolean_schema():
    json_schema = JSONSchema(schema={'bool': {'type': 'boolean'}})
    assert isinstance(json_schema._widgets['bool'], pn.widgets.Checkbox)

def test_number_schema():
    json_schema = JSONSchema(schema={'number': {'type': 'number', 'minimum': 0, 'maximum': 3.14}})
    widget = json_schema._widgets['number']
    assert isinstance(widget, pn.widgets.FloatSlider)
    assert widget.start == 0
    assert widget.end == 3.14
    
def test_enum_schema():
    json_schema = JSONSchema(schema={'enum': {'type': 'enum', 'enum': ['A', 'B', 'C']}})
    widget = json_schema._widgets['enum']
    assert isinstance(widget, pn.widgets.Select)
    assert widget.options == ['A', 'B', 'C']
