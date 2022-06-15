import os

import param

from panel.widgets import IntSlider

from lumen.variables import Variable, Variables


def test_variables():
    vars = Variables.from_spec({
        'str'  : 'a',
        'int'  : 1,
        'float': 1.4,
        'list' : [1, 2, 3]
    })

    # Test initialization
    assert ({'float', 'int', 'list', 'name', 'str'} - set(vars.param)) == set()

    assert vars.str == 'a'
    assert vars.int == 1
    assert vars.float == 1.4
    assert vars.list == [1, 2, 3]

    # Test update
    vars._vars['str'].value = 'b'
    assert vars.str == 'b'


def test_env_variable():
    var = Variable.from_spec({'type': 'env', 'key': 'LUMEN_VAR', 'default': 'B'})
    assert var.value == 'B'

    os.environ['LUMEN_VAR'] = 'A'
    var = Variable.from_spec({'type': 'env', 'key': 'LUMEN_VAR', 'default': 'B'})
    assert var.value == 'A'


def test_resolve_widget_variable_by_clsname():
    var = Variable.from_spec({'type': 'widget', 'kind': 'IntSlider'})

    assert isinstance(var._widget, IntSlider)


def test_widget_variable_label():
    var = Variable.from_spec({'type': 'widget', 'kind': 'IntSlider', 'label': 'Custom label'})

    assert isinstance(var._widget, IntSlider)
    assert var._widget.name == 'Custom label'


def test_resolve_widget_variable_by_module_ref():
    var = Variable.from_spec({'type': 'widget', 'kind': 'panel.widgets.IntSlider'})

    assert isinstance(var._widget, IntSlider)


def test_widget_variable_linking_unthrottled():
    var = Variable.from_spec({'type': 'widget', 'kind': 'IntSlider', 'throttled': False})

    var._widget.value = 3
    assert var.value == 3

    var.value = 4
    assert var._widget.value == 4


def test_widget_variable_linking_throttled():
    var = Variable.from_spec({'type': 'widget', 'kind': 'IntSlider'})

    with param.edit_constant(var._widget):
        var._widget.value_throttled = 3

    assert var.value == 3

    var.value = 4
    assert var._widget.value == 4
