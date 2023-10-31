import pytest

from lumen.sources.base import Source
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "spec,context,msg",
    (
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            {"sources": {}},
            None,
        ),
        (
            {"type": "file", "tables": {"penguins": "url.csv"}},
            None,
            None,
        ),
        (
            {"tables": {"penguins": "url.csv"}},
            None,
            "Source component specification did not declare a type",
        ),
        (
            {"type": "afile", "tables": {"penguins": "url.csv"}},
            None,
            "Source component specification declared unknown type 'afile'",
        ),
        (
            "src",
            {"sources": {"src": "Test"}},
            None,
        ),
        (
            "no_src",
            {"sources": {"src": "Test"}},
            "Referenced non-existent source 'no_src'",
        ),
    ),
    ids=[
        "correct1",
        "correct2",
        "missing_type",
        "wrong_type",
        "correct_string_spec",
        "wrong_string_spec",
    ],
)
def test_source_Source(spec, context, msg):
    if msg is None:
        if isinstance(spec, str):
            assert Source.validate(spec, context) == spec
        else:
            assert Source.validate(spec.copy(), context) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Source.validate(spec, context)


def test_source_key_validation():
    with pytest.raises(ValidationError, match="FileSource component specification contained unknown key 'tablas'"):
        Source.validate({'type': 'file', 'tablas': {}})
