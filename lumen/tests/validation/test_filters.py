import pytest

from lumen.filters import Filter
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "spec,msg",
    (
        (
            {"type": "constant", "field": "island", "value": "Torgersen"},
            None,
        ),
        (
            {"field": "island", "value": "Torgersen"},
            "Filter component specification did not declare a type",
        ),
        (
            {"type": "constant", "value": "Torgersen"},
            "The ConstantFilter component requires 'field' parameter to be defined",
        ),
        (
            {"type": "widget", "fields": "island"},
            "The WidgetFilter component requires 'field' parameter to be defined",
        ),
    ),
    ids=["correct1", "missing_type", "missing_field_constant", "missing_field_widget"],
)
def test_filter_filter(spec, msg):
    if msg is None:
        Filter.validate(spec)

    else:
        with pytest.raises(ValidationError, match=msg):
            Filter.validate(spec)
