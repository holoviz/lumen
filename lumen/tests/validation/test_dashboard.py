import pytest

from lumen.dashboard import Dashboard
from lumen.validation import ValidationError


@pytest.mark.parametrize(
    "sources,targets,msg",
    (
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "views": []}],
            None,
        ),
        (
            {"penguin": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "views": []}],
            "Target specified non-existent source 'penguins'",
        ),
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguin", "views": []}],
            "Target specified non-existent source 'penguin'",
        ),
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "view": []}],
            "Target component specification contained unknown key 'view'",
        ),
        (
            {"penguins": {"tables": {"penguins": "url.com"}}},
            [{"title": "Table", "source": "penguins", "views": []}],
            "Source component specification did not declare a type",
        ),
        (
            {"penguins": {"type": "file", "tables": {"penguins": "url.com"}}},
            [{"source": "penguins", "views": []}],
            "The Target component requires 'title' parameter to be defined",
        ),
    ),
    ids=[
        "correct",
        "missing_target1",
        "missing_target2",
        "missing_view",
        "missing_type",
        "missing_title",
    ],
)
def test_dashboard_Dashboard(sources, targets, msg):
    spec = {"sources": sources, "targets": targets}

    if msg is None:
        assert Dashboard.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            Dashboard.validate(spec)
