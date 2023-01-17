import pytest

from panel.template.base import BasicTemplate

from lumen.dashboard import Config


class CustomTemplate(BasicTemplate):
    pass

def no_arg_fn():
    return

def one_arg_fn(arg):
    return

def test_config_template():
    config = Config.from_spec({'template': 'lumen.tests.test_config.CustomTemplate'})
    assert config.template is CustomTemplate

def test_config_callbacks():
    config = Config.from_spec({
        'on_session_created': 'lumen.tests.test_config.no_arg_fn',
        'on_session_destroyed': 'lumen.tests.test_config.one_arg_fn',
        'on_loaded': 'lumen.tests.test_config.no_arg_fn',
        'on_error': 'lumen.tests.test_config.one_arg_fn',
        'on_update': 'lumen.tests.test_config.one_arg_fn',
    })
    assert config.on_session_created is no_arg_fn
    assert config.on_session_destroyed is one_arg_fn
    assert config.on_loaded is no_arg_fn
    assert config.on_error is one_arg_fn
    assert config.on_update is one_arg_fn

def test_config_callback_validation():
    with pytest.raises(ValueError, match=r'on_session_created callback must have signature func\(\), got func\(arg\)') as e:
        Config.from_spec({
            'on_session_created': 'lumen.tests.test_config.one_arg_fn',
        })
        assert e is None
