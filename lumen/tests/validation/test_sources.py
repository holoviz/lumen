import pytest

from lumen.sources import Source
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "spec,context,msg",
    (
        (
            {'type': 'file', 'tables': {'penguins': 'url.csv'}},
            {'sources': {}},
            None,
        ),
        (
            {'type': 'file', 'tables': {'penguins': 'url.csv'}},
            None,
            None,
        ),
        (
            {'tables': {'penguins': 'url.csv'}},
            None,
            'Source component specification did not declare a type',
        ),
        (
            {'type': 'afile', 'tables': {'penguins': 'url.csv'}},
            None,
            "Source component specification declared unknown type 'afile'",
        ),
        (
            'src',
            {'sources': {'src': 'Test'}},
            None,
        ),
        (
            "no_src",
            {'sources': {'src': 'Test'}},
            "Referenced non-existent source 'no_src'",
        ),


    ),
    ids=[
        "correct1", "correct2", "missing_type", "wrong_type",
        "correct_string_spec", "wrong_string_spec"
    ],
)
def test_filter_filter(spec, context, msg):
    if msg is None:
        Source.validate(spec, context)

    else:
        with pytest.raises(ValidationError, match=msg):
            Source.validate(spec, context)
