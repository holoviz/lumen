# Model Prompts

Some `Agent`s employ structured Pydantic models as their response format so that the code easily use the responses. These models can also be overridden by specifying the `model` key in `prompts`.

For example, the `SQLAgent`'s `main` default model is:

```python
class Sql(BaseModel):

    chain_of_thought: str = Field(
        description="""
        You are a world-class SQL expert, and your fame is on the line so don't mess up.
        Then, think step by step on how you might approach this problem in an optimal way.
        If it's simple, just provide one sentence.
        """
    )

    expr_slug: str = Field(
        description="""
        Give the SQL expression a concise, but descriptive, slug that includes whatever transforms were applied to it,
        e.g. top_5_athletes_gold_medals
        """
    )

    query: str = Field(description="Expertly optimized, valid SQL query to be executed; do NOT add extraneous comments.")
```

To override the `chain_of_thought` field, you can subclass the `Sql` model:

```python
from lumen.ai.models import Sql

class CustomSql(Sql):
    chain_of_thought: str = Field(
        description="Think through the query like an expert DuckDB user."
    )
```

Then, you can specify the `model` key in `prompts`:

```python
prompts = {
    "main": {
        "model": CustomSql
    }
}
agents = [lmai.agents.SQLAgent(prompts=prompts)]
ui = lmai.ExplorerUI(agents=agents)
```

Note, the field names in the model must match the original model's field names, or else unexpected fields will not be used.
