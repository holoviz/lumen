# :material-memory: Context

Context is shared memory between agents. When SQLAgent creates a pipeline, it adds it to context. When AnalystAgent runs, it reads that pipeline from context.

**See also:** [Agents](agents.md) — Agents communicate through context using input and output schemas.

## How context works

Agents communicate by reading from and writing to a shared dictionary:

``` py title="Agent adds to context"
async def respond(self, messages, context, **kwargs):
    # Do work
    result = "Sales increased 20%"
    
    # Return what to show user + what to add to context
    return [result], {"summary": result}  # (1)!
```

1. Context update gets merged into shared memory

Other agents can then access `context["summary"]`.

## Define requirements

Agents declare what they need using schemas:

``` py title="Input and output schemas" hl_lines="5-6 9-10"
from lumen.ai.context import ContextModel
from typing import NotRequired

class MyInputs(ContextModel):
    pipeline: object  # Required - agent won't run without this
    sql: NotRequired[str]  # Optional

class MyOutputs(ContextModel):
    summary: str  # Agent adds this to context
    metrics: dict
```

Fields without `NotRequired` are required. The agent only runs when required fields exist in context.

## Common context keys

| Key | Added by | Used by |
|-----|----------|---------|
| `source` | Initial setup | SQLAgent |
| `pipeline` | SQLAgent | VegaLiteAgent |
| `sql` | SQLAgent | ChatAgent |
| `data` | SQLAgent | ChatAgent | VegaLiteAgent
| `metaset` | TableListAgent | SQLAgent | ChatAgent |

## Safe context access

Always use `.get()` for optional keys:

``` py title="Check before accessing"
# Bad - crashes if missing
analysis = context['analysis']  # ❌

# Good - returns None if missing
analysis = context.get('analysis')  # ✅
if analysis is None:
    return ["No analysis yet"], {}
```

## Examples

### Agent requiring previous results

``` py title="Requires analysis from AnalystAgent" hl_lines="5-6"
from lumen.ai.context import ContextModel

class ReportInputs(ContextModel):
    pipeline: object
    analysis: str  # Must exist before agent runs
    
async def respond(self, messages, context, **kwargs):
    pipeline = context['pipeline']
    analysis = context['analysis']
    
    report = f"Results: {len(pipeline.data)} rows\n\n{analysis}"
    return [report], {"report": report}
```

### Tool providing context

``` py title="Tool adds to context" hl_lines="3-4"
from lumen.ai.tools import FunctionTool

tool = FunctionTool(
    function=calculate_totals,
    requires=["pipeline"],
    provides=["total_sales"]  # (1)!
)
```

1. Other agents can access `context["total_sales"]`

### Accumulating values

Collect values from multiple agents into a list:

``` py title="Accumulate sources" hl_lines="4"
from typing import Annotated

class MyInputs(ContextModel):
    sources: Annotated[list[object], ("accumulate", "source")]  # (1)!
```

1. Gathers all `source` values into `sources` list

If Agent A adds `{"source": s1}` and Agent B adds `{"source": s2}`, your agent sees `sources: [s1, s2]`.

## Best practices

**Use NotRequired for optional fields.** Most fields should be optional - only require what's absolutely necessary.

**Use meaningful key names.** `sales_summary` is better than `data`.

**Don't pollute context.** Only add keys other agents might use. Skip temporary/internal values.

**Check before accessing.** Always use `.get()` for optional keys.

**Document your schemas.** Add Field descriptions to help others understand what the data represents.

## Debug context

Enable debug logging to see context updates:

``` py
ui = lmai.ExplorerUI(data='penguins.csv', log_level='DEBUG')
```

Check console output for context keys and values.
