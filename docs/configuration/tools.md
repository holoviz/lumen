# :material-tools: Tools

**Tools let agents access external data and perform specialized tasks.**

Most users don't need custom tools. Built-in tools handle common needs.

**See also:** [Agents](agents.md) — Agents invoke tools when needed. For more complex logic, you may want to create a custom agent instead.

## Built-in tools

Lumen includes tools automatically:

- **TableLookup** - Finds relevant tables in your data (see [Vector Stores](vector_stores.md#table-discovery))
- **DocumentLookup** - Searches uploaded documents (see [Vector Stores](vector_stores.md#document-search))
- **DbtslLookup** - Queries dbt Semantic Layer metrics

You don't need to configure these. Agents use them when needed.

## Create a simple tool

If you require a custom tool, e.g. either to provide additional context, render some output or perform some action simply provide a function with type annotations and a docstring:

``` py title="Simple function tool"
import lumen.ai as lmai

def calculate_average(numbers: list[float]) -> float:
    """
    Calculate the average of a list of numbers.

    Parameters
    ----------
    numbers : list[float]
        Numbers to average

    Returns
    -------
    float
        The average
    """
    return sum(numbers) / len(numbers)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[calculate_average]  # (1)!
)
ui.servable()
```

1. Function automatically becomes a tool - the LLM uses your docstring and type hints

Lumen can now call this function by filling in the arguments. The return value is surfaced to the model, and only added to context if `provides` is set.

## Define a tool with metadata

Some tools require access to the current context, e.g. to access the current data. To declare that a particular argument should be looked up in the context you can use the `define_tool` decorator to annotate the function, ensuring the `FunctionTool` can populate `requires`, `provides`, and `purpose`.

As an example we can define a function that accept the `pipeline` and counts the number of rows in the table:

``` py title="Tool annotations"
import lumen.ai as lmai
from lumen.ai.tools import define_tool

@define_tool(
    requires=["pipeline"],
    purpose="Count rows in the active table"
)
def count_rows(pipeline) -> int:
    """Count total rows in the current table."""
    return len(pipeline.data)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[count_rows]
)
ui.servable()
```

## Render tool output

If your tool returns a value you want to render directly, set `render_output=True`:

``` py title="Render tool output"
import lumen.ai as lmai
from panel_material_ui import Card
from lumen.ai.tools import define_tool

@define_tool(render_output=True, purpose="Show a greeting card")
def greeting() -> Card:
    return Card(
        "Hello from Lumen tools!",
        title="Greeting",
        collapsed=True
    )

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[greeting]
)
ui.servable()
```

## Explicit `FunctionTool` definition

You may also explicitly define a `FunctionTool` instance:

``` py title="Tool with context access" hl_lines="26-28"
from lumen.ai.tools import FunctionTool

def filter_penguins(table) -> dict:
    """
    Filter penguins by bill length.

    Parameters
    ----------
    table : pd.DataFrame
        The penguin data

    Returns
    -------
    dict
        Filtered data and summary
    """
    filtered = table[table['bill_length_mm'] > 40]
    return {
        "filtered_table": filtered,
        "summary": f"Found {len(filtered)} penguins with bill length > 40mm"
    }

tool = FunctionTool(
    function=filter_penguins,
    requires=["table"],              # (1)!
    provides=["filtered_table", "summary"],  # (2)!
    purpose="Filter penguins by bill length"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
ui.servable()
```

1. Tool reads `table` from context
2. Tool adds `filtered_table` and `summary` to context (function must return a dict with those keys)

## Tool that calls an API

Wrap external services:

``` py title="API tool"
def fetch_weather(location: str) -> str:
    """
    Get current weather for a location.

    Parameters
    ----------
    location : str
        City name

    Returns
    -------
    str
        Weather description
    """
    import requests
    response = requests.get(f"https://api.weather.gov/...")
    return f"Weather: {response.json()['temp']}°F"

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[fetch_weather]
)
ui.servable()
```

## Complete example: Data validation

``` py title="Data quality tool" linenums="1"
import pandas as pd
from lumen.ai.tools import FunctionTool

def validate_quality(table: pd.DataFrame) -> dict:
    """
    Check data quality and report issues.

    Parameters
    ----------
    table : pd.DataFrame
        Data to validate

    Returns
    -------
    dict
        Validation report
    """
    missing = table.isnull().sum().sum()
    duplicates = table.duplicated().sum()

    issues = []
    if missing > 0:
        issues.append(f"{missing} missing values")
    if duplicates > 0:
        issues.append(f"{duplicates} duplicate rows")

    return {
        "total_rows": len(table),
        "issues": issues,
        "status": "✓ Clean" if not issues else "⚠️ Issues found"
    }

tool = FunctionTool(
    function=validate_quality,
    requires=["table"],
    provides=["data_quality_report"],
    purpose="Validate data quality and report issues"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
ui.servable()
```

## Tool components

**`requires`** - Context keys the tool needs:

``` py
requires=["table", "sql"]  # Tool receives these from context
```

**`provides`** - Context keys the tool creates (for a single key, a non-dict return value is wrapped):

``` py
provides=["summary", "report"]  # Tool adds these to context
```

**`purpose`** - Description for the LLM:

``` py
purpose="Validates data quality and finds issues"
```

## Multiple tools

Combine tools for complex workflows:

=== "With built-in tools"

    ``` py title="Mix custom and built-in tools"
    from lumen.ai.tools import DocumentLookup

    def get_stats(table) -> dict:
        """Calculate summary statistics."""
        return {
            "min": table['bill_length_mm'].min(),
            "max": table['bill_length_mm'].max(),
            "mean": table['bill_length_mm'].mean(),
        }

    def filter_species(table, species: str) -> dict:
        """Filter by species name."""
        filtered = table[table['species'] == species]
        return {
            "filtered": filtered,
            "count": len(filtered)
        }

    ui = lmai.ExplorerUI(
        data='penguins.csv',
        tools=[get_stats, filter_species, DocumentLookup()]
    )
    ui.servable()
    ```

=== "Only custom tools"

    ``` py title="Multiple custom tools"
    def tool_a(data: list) -> dict:
        """Process data."""
        return {"result_a": processed}

    def tool_b(data: list) -> dict:
        """Analyze data."""
        return {"result_b": analyzed}

    ui = lmai.ExplorerUI(
        data='penguins.csv',
        tools=[tool_a, tool_b]
    )
    ui.servable()
    ```

## Best practices

### Write clear docstrings

``` py title="Good docstring format"
def my_tool(data: list) -> str:
    """
    One-line summary of what it does.

    Detailed explanation if needed.

    Parameters
    ----------
    data : list
        What the data represents

    Returns
    -------
    str
        What gets returned
    """
```

### Use type hints

``` py title="Type hints help the LLM"
def process(numbers: list[float], threshold: int) -> dict:
    """Type hints help the LLM call correctly."""
```

### Name parameters clearly

``` py
# Good
def calculate_average(numbers: list[float]) -> float:

# Bad
def calculate(x: list[float]) -> float:
```

### Keep tools focused

``` py
# Good - one task
def validate_email(email: str) -> bool:

# Bad - too many tasks
def validate_and_process_user_data(data: dict):
```

### Return structured data when using provides

!!! warning "Match provides with return keys"

    When using `provides`, your function must return a dict with those keys:

    ``` py hl_lines="3 7-10"
    tool = FunctionTool(
        function=my_function,
        provides=["result", "metadata"]  # These keys must be in return dict
    )

    def my_function(data):
        return {
            "result": processed_data,
            "metadata": {"count": 10}
        }  # ✅ Has both "result" and "metadata"
    ```

### Handle errors gracefully

=== "Return error dict"

    ``` py title="Structured error handling"
    def process(data: list) -> dict:
        if not data:
            return {"error": "No data provided"}

        try:
            result = sum(data) / len(data)
            return {"average": result}
        except Exception as e:
            return {"error": str(e)}
    ```

=== "Return error string"

    ``` py title="Simple error handling"
    def process(data: list) -> str:
        if not data:
            return "Error: No data provided"

        try:
            result = sum(data) / len(data)
            return f"Average: {result:.2f}"
        except Exception as e:
            return f"Error: {e}"
    ```

## MCP tools

[MCP (Model Context Protocol)](https://modelcontextprotocol.io/) lets you connect to external tool servers. `MCPTool` wraps an MCP server tool and makes it available inside Lumen, working just like a `FunctionTool`.

### Connect to an MCP server

Use `MCPTool.from_server` to discover all tools on a server and create one `MCPTool` per tool:

``` py title="Discover tools from an MCP server"
import asyncio
import lumen.ai as lmai
from lumen.ai.tools import MCPTool

tools = asyncio.run(MCPTool.from_server("https://my-mcp-server.example.com/mcp"))

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=tools
)
ui.servable()
```

You can also connect to a local Python MCP server:

``` py title="Local MCP server"
tools = asyncio.run(MCPTool.from_server("my_mcp_server.py"))
```

Or pass a `fastmcp.FastMCP` instance directly (useful for testing):

``` py title="In-process MCP server"
from fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

tools = asyncio.run(MCPTool.from_server(mcp))
```

### Create a single MCPTool manually

If you already know the tool name and schema, you can create an `MCPTool` directly:

``` py title="Manual MCPTool"
from lumen.ai.tools import MCPTool

tool = MCPTool(
    server="https://my-mcp-server.example.com/mcp",
    tool_name="add",
    schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    description="Add two numbers",
)
```

## Two ways to use tools

Lumen supports tools in two distinct modes. Understanding the difference is important for choosing the right approach.

### Coordinator tools

Tools passed to `ExplorerUI(tools=...)` are **coordinator tools**. The coordinator (planner) decides when to invoke them as part of a multi-step plan. These tools participate in the full agent orchestration:

- The coordinator sees each tool's `purpose`, `requires`, and `provides`
- It selects tools based on what the user asked and what context is available
- Tool results flow into working memory and can be used by subsequent steps

``` py title="Coordinator tools"
import lumen.ai as lmai
from lumen.ai.tools import MCPTool

tools = asyncio.run(MCPTool.from_server(mcp_server))

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=tools  # (1)!
)
```

1. The coordinator orchestrates these tools alongside built-in agents

This is the right choice when tools need to interoperate with Lumen's data pipeline, agents, and context system.

### LLM tools (native tool calling)

Tools passed to `Llm(tools=...)` are **LLM tools**. They are sent as tool definitions directly to the LLM provider's API (OpenAI, Anthropic, Google, etc.) and the model decides when to call them using its built-in tool-calling capability:

- The LLM sees the tool schema and decides whether to call it
- The LLM fills in the arguments from the conversation context
- Results are returned to the LLM as follow-up messages
- This bypasses the coordinator and agent system entirely

``` py title="LLM tools"
import asyncio
import lumen.ai as lmai
from lumen.ai.tools import MCPTool

tools = asyncio.run(MCPTool.from_server(mcp_server))

llm = lmai.llm.Google(
    tools=tools  # (1)!
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    llm=llm
)
```

1. The LLM itself decides when to call these tools during any conversation turn

LLM tools are also combined with any tools passed per-call:

``` py title="Per-call tools combined with instance tools"
# Instance tools are always available
llm = lmai.llm.OpenAI(tools=[always_available_tool])

# Per-call tools are added on top for this specific call
result = await llm.invoke(messages, tools=[extra_tool])
```

Both `FunctionTool` and `MCPTool` instances work in either mode.

### Choosing between coordinator and LLM tools

| Coordinator tools (`ExplorerUI(tools=...)`) | LLM tools (`Llm(tools=...)`) |
|---------------------------------------------|------------------------------|
| Orchestrated by the planner as part of a multi-step plan | Called directly by the LLM during a single turn |
| Can read from and write to Lumen's working memory (`requires`/`provides`) | Operate independently of Lumen's context system |
| Best for tools that integrate with data pipelines | Best for self-contained tools (APIs, calculations) |
| Planner decides when the tool is relevant | LLM decides when to call based on conversation |

## When to use tools vs agents

| Use tools when | Use agents when |
|----------------|-----------------|
| Simple function call | Complex prompting needed |
| No async/await needed | Multiple LLM calls required |
| Wrapping external API | Multi-step reasoning needed |
| Straightforward logic | Sophisticated error handling |

## Troubleshooting

### Tool never gets called

The coordinator doesn't think the tool is relevant. Make the `purpose` clear and specific:

``` py
# Bad
purpose = "Does stuff with data"

# Good
purpose = "Validates email addresses and returns True if valid"
```

### Missing required argument

Tool expects a context key that doesn't exist. Ensure `requires` lists correct keys:

``` py
tool = FunctionTool(
    function=my_function,
    requires=["table"],  # Must exist in context
)
```

### Tool fails silently

Add error handling and return error messages instead of raising exceptions:

``` py
def my_tool(data):
    try:
        # Your logic
        return result
    except Exception as e:
        return {"error": str(e)}
```

### KeyError when using provides

!!! failure "Common mistake with provides"

    Your function must return a dict with all keys listed in `provides`:

    ``` py hl_lines="2 5 8 11-14"
    # Wrong - returns a string but provides expects dict keys
    provides=["summary", "count"]

    def bad_tool(data):
        return "Summary text"  # ❌

    # Correct - returns dict with expected keys
    provides=["summary", "count"]

    def good_tool(data):
        return {
            "summary": "Summary text",
            "count": len(data)
        }  # ✅
    ```

## See also

- [Vector Stores](vector_stores.md) - Configure document search and table discovery tools
- [Embeddings](embeddings.md) - Configure semantic search for tools
- [Agents](agents.md) - When to use agents instead of tools
- [MCP specification](https://modelcontextprotocol.io/) - The Model Context Protocol standard
- [fastmcp](https://gofastmcp.com/) - Python library for building and connecting to MCP servers
