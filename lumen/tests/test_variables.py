import os

import param  # type: ignore

from panel.widgets import IntSlider, TextInput

from lumen.base import Component
from lumen.state import state
from lumen.variables import (
    Parameter, Variable, Variables, Widget,
)


class CustomComponent(Component):

    string = param.String()

    number = param.Number()

    dictionary = param.Dict()


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

def test_add_variable_parameter():
    vars = Variables()

    c = CustomComponent(number=3.14)
    var = vars.add_variable(c.param.number)

    assert isinstance(var, Parameter)
    assert var.name == 'number'
    assert var.value == 3.14

    c.number *= 2

    assert var.value == 6.28

def test_add_variable_widget():
    vars = Variables()

    w = TextInput(name='text', value='foo')
    var = vars.add_variable(w)

    assert isinstance(var, Widget)
    assert var.name == 'text'
    assert var.value == 'foo'

    w.value = 'bar'

    assert var.value == 'bar'

def test_add_variable_widget_parameter():
    vars = Variables()

    w = IntSlider(name='int', value=3)
    var = vars.add_variable(w.param.value_throttled)

    assert isinstance(var, Widget)
    assert var.name == 'int'
    assert var.throttled
    assert var.value == 3

    w.value = 6

    assert var.value == 3

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

def test_widget_variable_intslider_to_spec():
    input_spec = {'type': 'widget', 'kind': 'IntSlider', 'throttled': False, 'start': 10, 'end': 53}
    excepted_spec = {'value': 10, **input_spec}
    output_spec = Variable.from_spec(input_spec).to_spec()
    assert excepted_spec == output_spec

def test_widget_variable_select_to_spec():
    input_spec = {'type': 'widget', 'kind': 'Select', 'options': ['A', 'B', 'C']}
    excepted_spec = {'value': 'A', **input_spec}
    output_spec = Variable.from_spec(input_spec).to_spec()
    assert excepted_spec == output_spec

def test_intslider_value_initialize():
    var = Widget(kind="IntSlider", value=20)
    assert var.value == 20

def test_variable_resolve_simple_var():
    state._variable = vars = Variables.from_spec({
        'str'  : 'a',
        'int'  : 1,
    })
    obj = CustomComponent(refs={'string': "$variables.str"})
    assert obj.string == 'a'
    vars.str = 'b'
    assert obj.string == 'b'

def test_variable_resolve_str_expr():
    state._variable = vars = Variables.from_spec({
        'str'  : 'a',
        'int'  : 1,
    })
    obj = CustomComponent(refs={'string': "'{0}_{1}'.format($variables.str, $variables.int)"})
    assert obj.string == 'a_1'
    vars.int = 2
    assert obj.string == 'a_2'
    vars.str = 'b'
    assert obj.string == 'b_2'

def test_variable_resolve_numeric_expr():
    state._variable = vars = Variables.from_spec({
        'float': 3.1,
        'int'  : 1,
    })
    obj = CustomComponent(refs={'number': "$variables.int + $variables.float"})
    assert obj.number == 4.1
    vars.int = 2
    assert obj.number == 5.1
    vars.float = 1.2
    assert obj.number == 3.2

def test_variable_resolve_numeric_expr_in_dict():
    state._variable = vars = Variables.from_spec({
        'float': 3.1,
        'int'  : 1,
    })
    obj = CustomComponent(
        dictionary={'a': 4.1},
        refs={'dictionary.a': "$variables.int + $variables.float"}
    )
    assert obj.dictionary['a'] == 4.1
    vars.int = 2
    assert obj.dictionary['a'] == 5.1
    vars.float = 1.2
    assert obj.dictionary['a'] == 3.2
