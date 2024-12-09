import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.translate import doc_descriptions, function_to_model


def add(a: int, b: int) -> int:
    """
    Adds two integers

    Parameters
    ----------
    a : int
        Integer A
    b : int
        Integer B

    Returns
    -------
    int
        Result of addition
    """
    return a+b

def add_google(a: int, b: int) -> int:
    """
    Adds two integers

    Args:
        a (int): Integer A
        b (int): Integer B

    Returns:
        int: Result of addition
    """
    return a + b


def add_sphinx(a: int, b: int) -> int:
    """
    Adds two integers

    :param a: Integer A
    :type a: int
    :param b: Integer B
    :type b: int
    :return: The sum of the two integers.
    :rtype: int
    """
    return a + b


def test_doc_descriptions_numpy():
    main, descriptions = doc_descriptions(add)
    assert main == "Adds two integers"
    assert descriptions["a"] == "Integer A"
    assert descriptions["b"] == "Integer B"

def test_doc_descriptions_google():
    main, descriptions = doc_descriptions(add_google)
    assert main == "Adds two integers"
    assert descriptions["a"] == "Integer A"
    assert descriptions["b"] == "Integer B"

def test_doc_descriptions_sphinx():
    main, descriptions = doc_descriptions(add_sphinx)
    assert main == "Adds two integers"
    assert descriptions["a"] == "Integer A"
    assert descriptions["b"] == "Integer B"

def test_function_to_model_args():
    model = function_to_model(add)

    fields = model.model_fields
    assert len(fields) == 2
    assert "a" in fields
    assert fields["a"].annotation is int
    assert fields["a"].description == "Integer A"
    assert "b" in fields
    assert fields["b"].annotation is int
    assert fields["b"].description == "Integer B"
