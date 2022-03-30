import param

from lumen.filters import Filter


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

    assert widget.value == (0, 2)

    with param.edit_constant(widget):
        widget.value_throttled = (1, 2)

    assert wfilter.value == (1, 2)

    wfilter.value = (2, 2)

    assert widget.value == (2, 2)
