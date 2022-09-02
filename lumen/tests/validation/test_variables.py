import pytest

from lumen.dashboard import Dashboard
from lumen.validation import ValidationError
from lumen.variables import Variable


def test_variables_validation():
    spec = {
        'variables': {
            'var1': {'type': 'constant', 'value': 'foo'},
            'var2': {'type': 'widget', 'value': '$variables.var1'}
        }
    }
    assert Dashboard.validate(spec) == spec
    assert Variable.validate(spec['variables']['var1'], runtime=True) == spec['variables']['var1']
    assert Variable.validate(spec['variables']['var2'], spec, runtime=True) == spec['variables']['var2']

def test_variables_wrong_type():
    spec = {
        'variables': [
            {'type': 'constant', 'value': 'foo'}
        ]
    }
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert 'Dashboard variables field expected dict type but got' in str(excinfo.value)

def test_variable_no_type():
    spec = {
        'variables': {
            'var1': {'value': 'foo'}
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert 'Variable component specification did not declare a type.' in str(excinfo.value)

def test_variable_wrong_parameter_type():
    spec = {
        'variables': {
            'var1': {'type': 'widget', 'value': 'foo', 'throttled': 'True'}
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert 'Widget component \'throttled\' value failed validation' in str(excinfo.value)

def test_variable_unknown_type():
    spec = {
        'variables': {
            'var1': {'type': 'foo', 'value': 'foo'}
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert 'Variable component declared unknown type \'foo\'.' in str(excinfo.value)

def test_variable_unknown_ref():
    spec = {
        'variables': {
            'var1': {'type': 'constant', 'value': '$variables.bar'}
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert 'Constant component \'value\' references undeclared variable \'$variables.bar\'' in str(excinfo.value)
