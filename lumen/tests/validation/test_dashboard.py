import pytest

from lumen.dashboard import Dashboard
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "sources,layouts,msg",
    (
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "views": []}],
            None,
        ),
        (
            {"penguin": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "views": []}],
            "Layout specified non-existent source 'penguins'",
        ),
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguin", "views": []}],
            "Layout specified non-existent source 'penguin'",
        ),
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "view": []}],
            "Layout component specification contained unknown key 'view'",
        ),
        (
            {"penguins": {"tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "views": []}],
            "Source component specification did not declare a type",
        ),
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"source": "penguins", "views": []}],
            "The Layout component requires 'title' parameter to be defined",
        ),
    ),
    ids=[
        "correct",
        "missing_layout1",
        "missing_layout2",
        "missing_view",
        "missing_type",
        "missing_title",
    ],
)
def test_dashboard_Dashboard(sources, layouts, msg):
    spec = {"sources": sources, "layouts": layouts}

    if msg is None:
        assert Dashboard.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Dashboard.validate(spec)


@pytest.mark.parametrize(
    "config,msg",
    (
        (
            {
                "theme": "dark",
                "layout": "column",
                "template": "bootstrap",
                "loading_color": "red",
            },
            None,
        ),
        (
            {"loading_color": 3},
            "Config component 'loading_color' value failed validation",
        ),
        (
            {"them": "dark"},
            "Config component specification contained unknown key 'them'",
        ),
        (
            {"theme": "purple"},
            "Config theme 'purple' could not be found",
        ),
        (
            {"template": "foobar"},
            "Config template 'foobar' not found",
        ),
        (
            {"template": "foobar.CustomTemplate"},
            "Config template 'foobar' module could not be imported errored with",
        ),
        (
            {"template": "lumen.CustomTemplate"},
            "Config template 'CustomTemplate' was not found in 'lumen' module",
        ),
        (
            {"template": "lumen.Dashboard"},
            "Config template 'lumen.Dashboard' is not a valid Panel template",
        ),
    ),
    ids=[
        "correct",
        "invalid_param_type",
        "misspell_theme",
        "invalid_theme",
        "invalid_template",
        "invalid_template_module",
        "invalid_template_cls_not_existing",
        "invalid_template_cls_wrong_type",
    ],
)
def test_dashboard_Config(config, msg):
    spec = {"config": config}

    if msg is None:
        assert Dashboard.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Dashboard.validate(spec)


@pytest.mark.parametrize(
    "defaults,msg",
    (
        (
            {
                "sources": [{"type": "file", "dask": True}],
                "filters": [{"type": "constant", "sync_with_url": False}],
                "transforms": [{"type": "sort", "ascending": False}],
                "views": [{"type": "table", "page_size": 10}],
            },
            None,
        ),
        (
            {"source": [{"type": "file", "dask": True}]},
            "Defaults component specification contained unknown key 'source'",
        ),
        (
            {"sources": [{"dask": True}]},
            "Defaults must declare the type the defaults apply to",
        ),
        (
            {"sources": [{"type": "file", "dsk": True}]},
            "Default for FileSource 'dsk' parameter cannot be set as there is no such parameter",
        ),
        (
            {"sources": [{"type": "file", "dask": "purple"}]},
            "The default for FileSource 'dask' parameter failed validation",
        ),
    ),
    ids=[
        "correct",
        "key_wrong",
        "no_type",
        "parameter_does_not_exist",
        "parameter_value_wrong_type",
    ],
)
def test_dashboard_Defaults(defaults, msg):
    spec = {"defaults": defaults}

    if msg is None:
        assert Dashboard.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Dashboard.validate(spec)


@pytest.mark.parametrize(
    "variables,msg",
    (
        (
            {
                "var1": {"type": "constant", "value": "foo"},
                "var2": {"type": "widget", "value": "$variables.var1", "kind": "TextInput"},
            },
            None,
        ),
        (
            [{"type": "constant", "value": "foo"}],
            "Dashboard component 'variables' key expected dict type but got list",
        ),
        (
            {"var1": {"value": "foo"}},
            "Variable component specification did not declare a type",
        ),
        (
            {"var1": {"type": "widget", "value": "foo", "throttled": "True", "kind": "TextInput"}},
            "Widget component 'throttled' value failed validation",
        ),
        (
            {"var1": {"type": "foo", "value": "foo"}},
            "Variable component specification declared unknown type 'foo'",
        ),
        (
            {"var1": {"type": "constant", "value": "$variables.bar"}},
            "Constant component 'value' references undeclared variable '\$variables",
        ),
    ),
    ids=[
        "correct",
        "wrong_type",
        "no_type",
        "wrong_parameter_type",
        "unknown_type",
        "unknown_ref",
    ],
)
def test_dashboard_Variables(variables, msg):
    spec = {"variables": variables}

    if msg is None:
        assert Dashboard.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Dashboard.validate(spec)
