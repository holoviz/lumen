import re

import pytest

from lumen.validation import ValidationError


def test_only_bold_whole_words():
    spec = {'title': 'Table', 'source': 'penguin', 'views': [{'type': 'table', 'table': 'penguins'}]}
    msg = 'test\n\n    title: Table\n    source: \x1b[1mpenguin\x1b[0m\n    views:\n    - type: table\n      table: penguins\n'
    with pytest.raises(ValidationError, match=re.escape(msg)):
        raise ValidationError("test", spec, "penguin")
