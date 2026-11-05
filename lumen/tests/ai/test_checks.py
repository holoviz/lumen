import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.checks import acheck
from lumen.ai.data import read
from lumen.tests.ai.conftest import MockLLM


@pytest.fixture
def penguins():
    return read({"penguins": pd.DataFrame({
        "species": ["Adelie", "Gentoo", "Adelie", "Chinstrap"], "body_mass_g": [3700, 5000, 1500, -1],
    })})


class RuleLLM(MockLLM):
    """Writes the query for a plain-language rule, and keeps the prompt."""

    query = "SELECT * FROM penguins WHERE body_mass_g < 2000"
    prompt = None

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        self.prompt = messages[-1]["content"]
        return response_model(query=self.query)


async def test_a_sql_rule_that_holds_passes(penguins):
    result = await acheck(penguins, "SELECT * FROM penguins WHERE body_mass_g > 100000")
    assert result and result.passed and result.count == 0
    assert repr(result) == "CheckResult(passed: SELECT * FROM penguins WHERE body_mass_g > 100000)"


async def test_a_sql_rule_that_breaks_returns_the_offending_rows(penguins):
    result = await acheck(penguins, "SELECT * FROM penguins WHERE body_mass_g <= 0")
    assert not result and result.count == 1
    assert result.rows.to_dict("records") == [{"species": "Chinstrap", "body_mass_g": -1}]


async def test_a_failed_check_fails_an_assert_with_its_rows(penguins):
    result = await acheck(penguins, "SELECT * FROM penguins WHERE body_mass_g < 2000")
    with pytest.raises(AssertionError, match=r"(?s)failed, 2 rows break.*Chinstrap"):
        result.raise_()
    with pytest.raises(AssertionError):
        assert result


@pytest.mark.parametrize("rule", ["DELETE FROM penguins", "DROP TABLE penguins", "SELECT 1; DELETE FROM penguins"])
async def test_a_check_cannot_change_the_data(penguins, rule):
    with pytest.raises(ValueError, match="read-only"):
        await acheck(penguins, rule)
    assert len(penguins.execute("SELECT * FROM penguins")) == 4


async def test_a_plain_language_rule_is_turned_into_sql_by_the_llm(penguins):
    llm = RuleLLM()
    result = await acheck(penguins, "no penguin weighs under 2 kg", llm=llm)
    assert result.sql == RuleLLM.query and result.count == 2
    assert "Rule: no penguin weighs under 2 kg" in llm.prompt and "body_mass_g (" in llm.prompt


async def test_a_comparison_is_a_plain_language_rule(penguins):
    llm = RuleLLM()
    await acheck(penguins, "body_mass_g > 0", llm=llm)
    assert llm.prompt is not None


async def test_an_llm_query_that_writes_is_refused(penguins):
    llm = RuleLLM()
    llm.query = "DELETE FROM penguins"
    with pytest.raises(ValueError, match="read-only"):
        await acheck(penguins, "drop the bad rows", llm=llm)


async def test_a_check_keeps_the_input_type():
    pl = pytest.importorskip("polars")
    source = read({"t": pl.DataFrame({"x": [1, -1]})})
    result = await acheck(source, "SELECT * FROM t WHERE x < 0")
    assert isinstance(result.rows, pl.DataFrame)


def test_source_check_works_in_a_plain_test(penguins):
    assert penguins.check("SELECT * FROM penguins WHERE species IS NULL")


async def test_a_plain_language_rule_sends_names_only_when_nothing_is_shared(penguins):
    from lumen.ai.answer import configure
    llm = RuleLLM()
    configure(share="none")
    try:
        await acheck(penguins, "no penguin weighs under 2 kg", llm=llm)
    finally:
        configure(share="sample")
    assert "penguins: species, body_mass_g" in llm.prompt and "(" not in llm.prompt.split("Tables and columns:")[1]
