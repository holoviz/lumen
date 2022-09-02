import pytest

from panel.layout import Column
from panel.template import BootstrapTemplate, DarkTheme

from lumen.dashboard import Config, Dashboard
from lumen.validation import ValidationError


def test_config():
    spec = {'config': {'theme': 'dark', 'layout': 'column', 'template': 'bootstrap', 'loading_color': 'red'}}
    assert Dashboard.validate(spec) == spec
    assert Config.validate(spec['config'], runtime=True) == {
        'theme': DarkTheme, 'layout': Column, 'template': BootstrapTemplate, 'loading_color': 'red'
    }

def test_config_invalid_param_type():
    spec = {'config': {'loading_color': 3}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config component 'loading_color' value failed validation" in str(excinfo.value)

def test_config_misspell_theme():
    spec = {'config': {'them': 'dark'}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config specification contained unknown field 'them'. Did you mean 'theme'?" in str(excinfo.value)

def test_config_invalid_theme():
    spec = {'config': {'theme': 'purple'}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config theme 'purple' could not be found. Theme must be one of" in str(excinfo.value)

def test_config_invalid_template():
    spec = {'config': {'template': 'foobar'}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config template 'foobar' not found. Template must be one of" in str(excinfo.value)

def test_config_invalid_template_module():
    spec = {'config': {'template': 'foobar.CustomTemplate'}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config template 'foobar' module could not be imported" in str(excinfo.value)

def test_config_invalid_template_cls_not_existing():
    spec = {'config': {'template': 'lumen.CustomTemplate'}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config template 'CustomTemplate' was not found in 'lumen' module." in str(excinfo.value)

def test_config_invalid_template_cls_wrong_type():
    spec = {'config': {'template': 'lumen.Dashboard'}}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Config template 'lumen.Dashboard' is not a valid Panel template." in str(excinfo.value)
