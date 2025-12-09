# Tools

**Tools let agents access external data and perform specialized tasks.**

Most users don't need custom tools. Built-in tools handle common needs.

## Built-in tools

Lumen includes tools automatically:

- **TableLookup** - Finds relevant tables in your data
- **DocumentLookup** - Searches uploaded documents
- **IterativeTableLookup** - Advanced table discovery
- **DbtslLookup** - Queries dbt Semantic Layer metrics

You don't need to configure these. Agents use them when needed.

## Create a simple tool

Turn any function into a tool:

```python
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
    tools=[calculate_average]
)
ui.servable()
```

The LLM uses your docstring and type hints to understand when to call it.

## Tool that accesses memory

Tools can read data from memory:

```python
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
    requires=["table"],              # Needs table from memory
    provides=["filtered_table", "summary"],  # Adds these to memory
    purpose="Filter penguins by bill length"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
```

## Tool that calls an API

Wrap external services:

```python
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
```

## Complete example: Data validation

```python
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
```

## Tool components

**`requires`** - Memory keys the tool needs:

```python
requires=["table", "sql"]  # Tool receives these from memory
```

**`provides`** - Memory keys the tool creates:

```python
provides=["summary", "report"]  # Tool adds these to memory
```

**`purpose`** - Description for the LLM:

```python
purpose="Validates data quality and finds issues"
```

## Multiple tools

Combine tools for complex workflows:

```python
from lumen.ai.tools import DocumentLookup

def get_stats(table) -> dict:
    """Calculate summary statistics."""
    return {
        "min": table['bill_length_mm'].min(),
        "max": table['bill_length_mm'].max(),
        "mean": table['bill_length_mm'].mean(),
    }

def filter_species(table, species: str):
    """Filter by species name."""
    return table[table['species'] == species]

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[get_stats, filter_species, DocumentLookup()]
)
```

## Best practices

**Write clear docstrings:**

```python
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

**Use type hints:**

```python
def process(numbers: list[float], threshold: int) -> dict:
    """Type hints help the LLM call correctly."""
```

**Name parameters clearly:**

```python
# Good
def calculate_average(numbers: list[float]) -> float:

# Bad
def calculate(x: list[float]) -> float:
```

**Keep tools focused:**

```python
# Good - one task
def validate_email(email: str) -> bool:

# Bad - too many tasks
def validate_and_process_user_data(data: dict):
```

**Handle errors gracefully:**

```python
def process(data: list) -> str:
    if not data:
        return "No data provided"
    
    try:
        result = sum(data) / len(data)
        return f"Average: {result:.2f}"
    except Exception as e:
        return f"Error: {e}"
```

## When to use tools vs agents

Use tools when:

- The task is a simple function call
- You don't need async/await
- You're wrapping an external API
- The logic is straightforward

Use agents when:

- You need complex prompting
- The task requires multiple LLM calls
- You need async operations
- The logic is sophisticated

## Troubleshooting

**Tool never gets called:**

Check that `purpose` is clear and specific. The LLM uses this to decide when to invoke the tool.

**"Missing required argument":**

Ensure `requires` lists the correct memory keys, and those keys exist when the tool runs.

**Tool fails silently:**

Add error handling and return error messages instead of raising exceptions:

```python
def my_tool(data):
    try:
        # Your logic
        return result
    except Exception as e:
        return f"Error: {e}"
```

**Tool returns unexpected data:**

Check type hints match what you're actually returning. The LLM relies on type hints to understand the output.
