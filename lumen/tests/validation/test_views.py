import pytest

from lumen.validation import ValidationError
from lumen.views.base import Download, View


@pytest.mark.parametrize(
    "spec,output",
    (
        (
            "csv",
            {"format": "csv", "type": "default"},
        ),
        (
            {"format": "csv"},
            {"format": "csv", "type": "default"},
        ),
        (
            {"formats": "csv"},
            "The Download component requires 'format' parameter to be defined",
        ),
        (
            {"format": "csvs"},
            "Download component 'format' value failed validation: csvs",
        ),
    ),
    ids=["correct1", "correct2", "missing_required", "wrong_format"],
)
def test_target_Download(spec, output):
    if isinstance(output, dict):
        if isinstance(spec, str):
            assert Download.validate(spec) == output
        else:
            assert Download.validate(spec.copy()) == output

    else:
        with pytest.raises(ValidationError, match=output):
            Download.validate(spec)


@pytest.mark.parametrize(
    "spec,msg",
    (
        (
            {"type": "download", "format": "csv"},
            None,
        ),
        (
            {"type": "download", "formats": "csv"},
            "The DownloadView component requires 'format' parameter to be defined",
        ),
        (
            {"type": "download", "format": "csvs"},
            "DownloadView component 'format' value failed validation: csvs",
        ),
        (
            {"format": "csv"},
            "View component specification did not declare a type",
        ),
        (
            {"type": "downloads", "format": "csv"},
            "View component specification declared unknown type 'downloads'",
        ),
    ),
    ids=["correct", "missing_required", "wrong_format", "missing_type", "wrong_type"],
)
def test_target_View(spec, msg):
    if msg is None:
        assert View.validate(spec.copy()) == spec

    else:
        with pytest.raises(ValidationError, match=msg):
            View.validate(spec)
