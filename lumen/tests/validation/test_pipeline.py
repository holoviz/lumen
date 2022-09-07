import pytest

from lumen.pipeline import Pipeline
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "source,filters,transforms,msg",
    (
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"type": "widget", "field": "species"}],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            None,
        ),
        (
            {"type": "files", "tables": {"penguins": "url.csv"}},
            [{"type": "widget", "field": "species"}],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            "Source component specification declared unknown type 'files'",
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"type": "widgets", "field": "species"}],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            "Filter component specification declared unknown type 'widgets'",
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"type": "widget", "field": "species"}],
            [{"type": "aggregat", "method": "mean", "by": ["species", "sex", "year"]}],
            "Transform component specification declared unknown type 'aggregat'",
        ),
        (
            {"tables": {"penguins": "url.csv"}},
            [{"type": "widget", "field": "species"}],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            "Source component specification did not declare a type",
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"field": "species"}],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            "Filter component specification did not declare a type",
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"type": "widget", "field": "species"}],
            [{"method": "mean", "by": ["species", "sex", "year"]}],
            "Transform component specification did not declare a type",
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"type": "widget", "fields": "species"}],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            "The WidgetFilter component requires 'field' parameter to be defined",
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [{"type": "widget", "field": "species"}],
            [],
            None,
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [],
            [{"type": "aggregate", "method": "mean", "by": ["species", "sex", "year"]}],
            None,
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            [],
            [],
            None,
        ),
    ),
    ids=[
        "correct1",
        "wrong_type1",
        "wrong_type2",
        "wrong_type3",
        "no_type1",
        "no_type2",
        "no_type3",
        "no_field",
        "no_transform",
        "no_filters",
        "no_filters_transforms",
    ],
)
def test_pipeline_Pipeline(source, filters, transforms, msg):
    spec = {
        "source": source,
        "filters": filters,
        "transforms": transforms,
    }
    if msg is None:
        assert Pipeline.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Pipeline.validate(spec)
