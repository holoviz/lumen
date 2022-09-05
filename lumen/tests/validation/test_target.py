import pytest

from lumen.target import Facet
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "spec,msg",
    (
        (
            {"by": ["model"], "layout": "column"},
            None,
        ),
        (
            {"bys": ["model"], "layout": "column"},
            "Facet component specification contained unknown key",
        ),
        (
            {"layout": "column"},
            "The Facet component requires 'by' parameter to be defined",
        ),
        (
            {"by": "model", "layout": "column"},
            "Facet component 'by' key expected list type but got str",
        ),
    ),
    ids=["correct", "unknown_key", "missing_required", "wrong_type"]
)
def test_target_facet(spec, msg):
    if msg is None:
        Facet.validate(spec)

    else:
        with pytest.raises(ValidationError, match=msg):
            Facet.validate(spec)
