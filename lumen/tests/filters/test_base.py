import param  # type: ignore

from panel.widgets import RangeSlider

from lumen.filters.base import Filter


def test_resolve_module_type():
    assert Filter._get_type('lumen.filters.base.Filter') is Filter


def test_widget_filter_link_unthrottled():
    wfilter = Filter.from_spec(
        {'type': 'widget', 'field': 'test', 'throttled': False},
        {'example': {
            'test': {
                'type': 'integer',
                'inclusiveMinimum': 0,
                'inclusiveMaximum': 2
            }
        }}
    )
    widget = wfilter.panel

    assert wfilter.query is None
    assert widget.value == (0, 2)

    widget.value = (1, 2)

    assert wfilter.value == (1, 2)

    wfilter.value = (2, 2)

    assert widget.value == (2, 2)

    widget.visible = False

    assert wfilter.visible == False

    widget.disabled = False

    assert widget.disabled == False


def test_widget_filter_link_throttled():
    wfilter = Filter.from_spec(
        {'type': 'widget', 'field': 'test'},
        {'example': {
            'test': {
                'type': 'integer',
                'inclusiveMinimum': 0,
                'inclusiveMaximum': 2
            }
        }}
    )
    widget = wfilter.panel

    assert wfilter.query is None
    assert widget.value == (0, 2)

    with param.edit_constant(widget):
        widget.value_throttled = (1, 2)

    assert wfilter.value == (1, 2)

    wfilter.value = (2, 2)

    assert widget.value == (2, 2)


def test_widget_filter_explicit_widget():
    wfilter = Filter.from_spec(
        {'type': 'widget', 'field': 'test', 'widget': 'panel.widgets.RangeSlider'},
        {'example': {
            'test': {
                'type': 'integer',
                'inclusiveMinimum': 1,
                'inclusiveMaximum': 3
            }
        }}
    )
    assert isinstance(wfilter.widget, RangeSlider)
    assert wfilter.widget.start == 1
    assert wfilter.widget.end == 3
    assert wfilter.query is None


def test_widget_filter_explicit_widget_to_spec():
    wfilter = Filter.from_spec(
        {'type': 'widget', 'field': 'test', 'widget': 'panel.widgets.RangeSlider'},
        {'example': {
            'test': {
                'type': 'integer',
                'inclusiveMinimum': 1,
                'inclusiveMaximum': 3
            }
        }}
    )
    assert wfilter.to_spec() == {'type': 'widget', 'field': 'test', 'widget': 'panel.widgets.slider.RangeSlider'}
