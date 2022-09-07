import pytest

from lumen.validation import ValidationError
from lumen.views import Download, View


@pytest.mark.parametrize(
    "spec,msg",
    (
        (
            "csv",
            None,
        ),
        (
            {'format': 'csv'},
            None,
        ),
        (
            {'formats': 'csv'},
            "The Download component requires 'format' parameter to be defined",
        ),
        (
            {'format': 'csvs'},
            "Download component 'format' value failed validation: csvs",
        )
    ),
    ids=["correct1", "correct2", "missing_required", "wrong_format"]
)
def test_target_download(spec, msg):
    if msg is None:
        Download.validate(spec)

    else:
        with pytest.raises(ValidationError, match=msg):
            Download.validate(spec)


@pytest.mark.parametrize(
    "spec,msg",
    (
        (
            {'type': 'download', 'format': 'csv'},
            None,
        ),
        (
            {'type': 'download', 'formats': 'csv'},
            "The DownloadView component requires 'format' parameter to be defined",
        ),
        (
            {'type': 'download', 'format': 'csvs'},
            "DownloadView component 'format' value failed validation: csvs",
        ),
        (
            {'format': 'csv'},
            "View component specification did not declare a type",
        ),
        (
            {'type': 'downloads', 'format': 'csv'},
            "View component specification declared unknown type 'downloads'",
        ),
    ),
    ids=["correct", "missing_required", "wrong_format", "missing_type", "wrong_type"]
)
def test_target_view(spec, msg):
    if msg is None:
        View.validate(spec)

    else:
        with pytest.raises(ValidationError, match=msg):
            View.validate(spec)
