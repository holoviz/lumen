import pytest

from lumen.transforms.base import Transform
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "spec,msg",
    (
        (
            {"type": "iloc", "end": 10},
            None,
        ),
        (
            {"type": "isloc", "end": 10},
            "Transform component specification declared unknown type 'isloc'",
        ),
        (
            {"end": 10},
            "Transform component specification did not declare a type",
        ),
    ),
    ids=["correct", "wrong_type", "missing_type"],
)
def test_transforms_Transform(spec, msg):
    if msg is None:
        assert Transform.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Transform.validate(spec)
