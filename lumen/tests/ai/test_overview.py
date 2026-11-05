import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from panel.tests.util import async_wait_until
from panel_material_ui import Button, FlexBox

from lumen.ai.answer import Answer, configure
from lumen.ai.data import read
from lumen.ai.overview import aoverview
from lumen.tests.ai.conftest import MockLLM


class QuestionsLLM(MockLLM):
    """Writes the suggested questions, and keeps the prompt it was sent."""

    prompt = None

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        self.prompt = messages[-1]["content"]
        return response_model(questions=[f"Question {i}?" for i in range(1, 6)])


@pytest.fixture
def messy_source():
    return read({"birds": pd.DataFrame({
        "weight": [1.0, 2.0, None, 4.0] * 10, "colony": ["north"] * 40, "wing": [10.5, 11.0, 12.5, 9.0] * 10,
    })})


async def test_overview_has_each_tables_columns_and_rows(messy_source):
    overview = await aoverview(messy_source, questions=False)
    assert overview.rows == {"birds": 40} and overview.sampled == {}
    columns = overview.columns["birds"].set_index("column")
    assert columns.loc["weight", "nulls (%)"] == 25.0 and columns.loc["colony", "distinct"] == 1
    assert columns.loc["wing", "max"] == 12.5


async def test_overview_flags_what_looks_wrong(messy_source):
    flags = " ".join((await aoverview(messy_source, questions=False)).flags["birds"])
    assert '"weight" (25.0% null)' in flags and 'Constant column(s) carrying no information: "colony"' in flags


async def test_overview_without_questions_makes_no_llm_call(messy_source):
    class NoCalls(MockLLM):
        async def invoke(self, *args, **kwargs):
            raise AssertionError("no call expected")

    overview = await aoverview(messy_source, questions=False, llm=NoCalls())
    assert overview.questions == [] and overview.usage is None


async def test_overview_asks_the_llm_for_five_questions(messy_source):
    llm = QuestionsLLM()
    overview = await aoverview(messy_source, llm=llm)
    assert overview.questions == [f"Question {i}?" for i in range(1, 6)]
    assert "Table birds (40 rows)" in llm.prompt and "weight,DOUBLE,1.0,4.0" in llm.prompt


async def test_overview_prompt_has_no_values_unless_samples_are_shared(messy_source):
    llm = QuestionsLLM()
    configure(share="schema")
    try:
        await aoverview(messy_source, llm=llm)
    finally:
        configure(share="sample")
    assert "weight,DOUBLE\n" in llm.prompt and "12.5" not in llm.prompt


async def test_clicking_a_question_asks_it_and_shows_the_answer(messy_source, monkeypatch):
    overview = await aoverview(messy_source, llm=QuestionsLLM())
    asked = []

    async def aask(question, **kwargs):
        asked.append(question)
        return Answer(question=question, value="Forty birds.", status="success")

    monkeypatch.setattr(messy_source, "aask", aask)
    panel = overview.__panel__()
    buttons = next(part for part in panel if isinstance(part, FlexBox))
    assert [b.label for b in buttons if isinstance(b, Button)] == overview.questions
    buttons[1].clicks += 1
    answers = panel[-1]
    await async_wait_until(lambda: len(answers) == 1)
    assert asked == ["Question 2?"] and answers[0][0].object == "Forty birds."


async def test_overview_outside_a_kernel_is_html(messy_source):
    overview = await aoverview(messy_source, llm=QuestionsLLM())
    bundle = overview._repr_mimebundle_()
    assert bundle["text/plain"] == "Overview(birds (40 rows); 3 flags, 5 questions)"
    assert "<h3>birds</h3>" in bundle["text/html"] and "<li>Question 5?</li>" in bundle["text/html"]


def test_source_overview_calls_the_llm_once(messy_source):
    llm = QuestionsLLM()
    overview = messy_source.overview(llm=llm)
    assert len(overview.questions) == 5 and str(overview).startswith("Overview(birds")


async def test_overview_says_when_its_statistics_are_from_a_sample(messy_source, monkeypatch):
    from lumen.ai import overview as overview_module
    from lumen.ai.table_stats import ColumnStats, TableStats

    class SampledStore:
        async def ensure(self, source, tables, timeout=None):
            stats = TableStats(table="birds", rows=10_000_000, sample_size=10_000, ranges_exact=False,
                               columns=[ColumnStats(name="weight", type="DOUBLE", nulls=0.0, distinct=50, min=1.0, max=4.0)])
            return {"birds": stats}

    monkeypatch.setattr(overview_module, "get_stats_store", SampledStore)
    overview = await aoverview(messy_source, questions=False)
    assert overview.sampled == {"birds": 10_000}
    assert "from a sample of 10,000 rows" in overview.__panel__()[0].object
    assert "from a sample of 10,000 rows" in overview._repr_mimebundle_()["text/html"]


async def test_overview_prompt_has_names_only_when_nothing_is_shared(messy_source):
    llm = QuestionsLLM()
    configure(share="none")
    try:
        await aoverview(messy_source, llm=llm)
    finally:
        configure(share="sample")
    assert llm.prompt == "Table birds:\nweight\ncolony\nwing"
