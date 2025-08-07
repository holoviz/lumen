from panel.layout import Column
from panel.pane import Markdown
from panel.widgets import Checkbox

from lumen.views.base import Panel, View


def test_panel_view_roundtrip():
    md = Markdown("TEST", width=420)

    spec = Panel(object=md).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'{md.name}',
            'object': 'TEST',
            'type': 'panel.pane.markup.Markdown',
            'width': 420,
        }
    }

    panel_view = Panel.from_spec(spec)
    obj = panel_view.object
    assert isinstance(obj, Markdown)
    assert obj.name == md.name
    assert obj.object == md.object
    assert obj.width == 420


def test_panel_layout_roundtrip():
    a, b = Markdown("A"), Markdown("B")
    column = Column(a, b)

    spec = Panel(object=column).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'{column.name}',
            'objects': [
                {'type': 'panel.pane.markup.Markdown', 'object': 'A', 'name': f'{a.name}'},
                {'type': 'panel.pane.markup.Markdown', 'object': 'B', 'name': f'{b.name}'}
            ],
            'type': 'panel.layout.base.Column'
        }
    }

def test_panel_cross_reference_param():
    a = Checkbox(name="A")
    b = Markdown("B", visible=a)
    column = Column(a, b)

    spec = Panel(object=column).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'{column.name}',
            'objects': [
                {
                    'type': 'panel.widgets.input.Checkbox',
                    'name': f'{a.name}'
                },
                {
                    'type': 'panel.pane.markup.Markdown',
                    'object': 'B',
                    'name': f'{b.name}',
                    'visible': {
                        'name': 'value',
                        'owner': 'A',
                        'type': 'param'
                    }
                }
            ],
            'type': 'panel.layout.base.Column'
        }
    }

def test_panel_cross_reference_rx():
    a = Checkbox(name="A")
    b = Markdown("B", visible=a.rx().rx.not_())
    column = Column(a, b)
    spec = Panel(object=column).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'{column.name}',
            'objects': [
                {
                    'type': 'panel.widgets.input.Checkbox',
                    'name': f'{a.name}'
                },
                {
                    'type': 'panel.pane.markup.Markdown',
                    'object': 'B',
                    'name': f'{b.name}',
                    'visible': {
                        'input_obj': None,
                        'kwargs': {},
                        'method': None,
                        'operation': {
                            'args': [],
                            'fn': '_operator.not_',
                            'kwargs': {},
                            'reverse': False,
                        },
                        'prev': {
                            'type': 'rx',
                            'input_obj': None,
                            'kwargs': {},
                            'method': None,
                            'operation': None,
                            'prev': {
                                'name': 'value',
                                'owner': 'A',
                                'type': 'param'
                            }
                        },
                        'type': 'rx',
                    }
                }
            ],
            'type': 'panel.layout.base.Column'
        }
    }
