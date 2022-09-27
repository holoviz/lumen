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
        (
            {'type': 'widget', 'widget': 'panel.widgets.IntSlider', 'field': 'year'},
            None
        ),
        (
            {'type': 'widget', 'widget': 'IntSlider', 'field': 'year'},
            "Filter could not resolve widget module reference 'IntSlider'."
        ),
    ),
    ids=["correct1", "missing_type", "missing_field_constant", "missing_field_widget", "valid_widget_ref", "wrong_widget_ref"],
)
def test_filter_Filter(spec, msg):
    if msg is None:
        assert Filter.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Filter.validate(spec)
