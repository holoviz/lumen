import pytest

from lumen.pipeline import Pipeline


def append_s_to_string(spec):
    """
    This function recursive add an extra "s" to each string in a nested dict/list
    """
    final = []
    items = []

    if isinstance(spec, list):
        items = enumerate(spec)

    if isinstance(spec, dict):
        for k in spec:
            s = spec.copy()
            if isinstance(k, str):
                s[k + "s"] = s[k]
                del s[k]
            final.append(s)

        items = spec.items()

    for k, v in items:
        s = spec.copy()
        if isinstance(v, str):
            s[k] = v + "s"
            final.append(s)
        elif isinstance(v, (dict, list)):
            red = append_s_to_string(v)
            for r in red:
                s2 = spec.copy()
                s2[k] = r
                final.append(s2)

    return final


def test_match_suggestion_message(penguins_file):
    spec = {
        "source": {"type": "file", "tables": {"penguins": penguins_file}},
        "filters": [
            {"type": "widget", "field": "species"},
        ],
        "transforms": [{"type": "aggregate", "method": "mean", "by": ["species"]}],
    }

    expected_error_msg = "(.+? Did you mean .+?|'type' for '.+?' is not available|Filter specification must declare field to filter on)"
    ignore_words = ["penguinss", "penguins.csvs", "speciess", "means"]

    for s in append_s_to_string(spec):
        ss = str(s)
        if any(iw in ss for iw in ignore_words):
            continue
        with pytest.raises(ValueError, match=expected_error_msg):
            Pipeline.from_spec(s)
