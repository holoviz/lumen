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

Turn any function into a tool:

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

## Tool that accesses context

Tools can read data from context (memory):

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
2. Tool adds `filtered_table` and `summary` to context

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

**`provides`** - Context keys the tool creates:

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
