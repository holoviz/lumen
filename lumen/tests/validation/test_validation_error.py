from lumen.validation import ValidationError


def test_only_bold_whole_words():
    spec = {'title': 'Table', 'source': 'penguin', 'views': [{'type': 'table', 'table': 'penguins'}]}
    msg = r'test\n\n    title: Table\n    source: \x1b[1mpenguin\x1b[0m\n    views:\n    - type: table\n      table: penguins\n'
    try:
        ValidationError("test", spec, "penguin")
    except Exception as e:
        assert e == msg
