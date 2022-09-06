import pytest

from lumen.dashboard import Dashboard
from lumen.validation import ValidationError


def test_defaults():
    spec = {
        'defaults': {
            'sources': [{'type': 'file', 'dask': True}],
            'filters': [{'type': 'constant', 'sync_with_url': False}],
            'transforms': [{'type': 'sort', 'ascending': False}],
            'views': [{'type': 'table', 'page_size': 10}],
        }
    }
    assert Dashboard.validate(spec) == spec

def test_defaults_key_wrong():
    spec = {'defaults': {
        'source': [{'type': 'file', 'dask': True}]
    }}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Defaults component specification contained unknown key 'source'. Did you mean 'sources'?" in str(excinfo.value)

def test_defaults_no_type():
    spec = {'defaults': {
        'sources': [{'dask': True}]
    }}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert "Defaults must declare the type the defaults apply to." in str(excinfo.value)

def test_defaults_parameter_does_not_exist():
    spec = {'defaults': {
        'sources': [{'type': 'file', 'dsk': True}]
    }}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert (
        "Default for FileSource 'dsk' parameter cannot be set as there is no such parameter. "
        "Did you mean 'dask'?"
    ) in str(excinfo.value)

def test_defaults_parameter_value_wrong_type():
    spec = {'defaults': {
        'sources': [{'type': 'file', 'dask': 'purple'}]
    }}
    with pytest.raises(ValidationError) as excinfo:
        Dashboard.validate(spec)
    assert (
        "The default for FileSource 'dask' parameter failed validation: Boolean parameter 'dask' "
        "must be True or False, not purple."
    ) in str(excinfo.value)
