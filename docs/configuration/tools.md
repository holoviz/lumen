# Tools

Tools extend agents' capabilities by enabling them to access external data, perform specialized tasks, and interact with their environment. Tools can query databases, search documents, look up tables, or perform domain-specific computations.

## Built-in tools

Lumen AI includes several built-in tools for common data exploration tasks:

**DocumentLookup**

- Searches uploaded documents using vector embeddings
- Finds relevant document chunks based on user queries
- Provides document context to agents

**TableLookup**

- Discovers relevant tables in your datasets
- Searches by table names and descriptions
- Provides context about available tables

**IterativeTableLookup**

- Advanced version of TableLookup
- Performs iterative selection of tables and schemas
- Provides complete SQL schemas for selected tables
- Best for complex data queries

**DbtslLookup**

- Integrates with dbt Semantic Layer
- Searches for relevant metrics and dimensions
- Queries business metrics for analysis

## Using tools

### Add tools to agents

Pass tools to ExplorerUI so agents can use them:

```python
import lumen.ai as lmai
from lumen.ai.tools import DocumentLookup, TableLookup

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[DocumentLookup(), TableLookup()]
)
ui.servable()
```

Agents automatically decide when to use available tools based on the query.

### Use function-based tools

Turn any Python function into a tool by adding it to the tools list:

```python
import lumen.ai as lmai

def calculate_average(data: list[float]) -> float:
    """
    Calculate the average of a list of numbers.
    
    Parameters
    ----------
    data : list[float]
        List of numbers to average
        
    Returns
    -------
    float
        The average value
    """
    return sum(data) / len(data)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[calculate_average]
)
ui.servable()
```

The LLM uses the function's docstring and type hints to understand when and how to call it.

### Control tool behavior with FunctionTool

For more control, wrap functions with `FunctionTool`:

```python
import lumen.ai as lmai
from lumen.ai.tools import FunctionTool

def apply_filter(table) -> str:
    """
    Applies a filter to the current table.
    
    Parameters
    ----------
    table : pd.DataFrame
        The data table
        
    Returns
    -------
    str
        Confirmation message
    """
    filtered = table[table['species'] == 'Adelie']
    return f"Filtered to {len(filtered)} Adelie penguins"

tool = FunctionTool(
    function=apply_filter,
    requires=["table"],              # Function needs 'table' from memory
    provides=[],                     # Function doesn't provide anything to memory
    purpose="Filters the table to show only Adelie penguins"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
ui.servable()
```

## Creating custom tools

Create custom tools for domain-specific tasks and specialized queries.

### Core concepts

**`requires`**

List of memory keys the tool needs. These are automatically injected as arguments:

```python
requires = ["table", "sql"]  # Tool will receive table and sql from memory
```

**`provides`**

List of memory keys the tool updates or creates:

```python
provides = ["summary"]  # Tool updates or creates the 'summary' key in memory
```

**`purpose`**

Clear description of what the tool does. Helps the LLM decide when to use it:

```python
purpose = "Filters data based on a specific column value"
```

**`formatter`**

Controls how the tool output is displayed. Default: `"{function}({arguments}) returned: {output}"`

```python
formatter = "Executed {function} with {arguments}, result: {output}"
```

### Simple function tool

Create a tool from a simple function:

```python
import lumen.ai as lmai

def summarize_data(data: list) -> str:
    """
    Summarize a list of data points.
    
    Returns the count and average.
    """
    if not data:
        return "No data available"
    return f"Count: {len(data)}, Average: {sum(data) / len(data):.2f}"

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[summarize_data]
)
ui.servable()
```

### Tool that accesses memory

Create tools that read from and write to memory:

```python
import lumen.ai as lmai
from lumen.ai.tools import FunctionTool

def apply_proprietary_algorithm(table) -> dict:
    """
    Applies a proprietary algorithm to filter the table.
    
    Parameters
    ----------
    table : pd.DataFrame
        The input data table
        
    Returns
    -------
    dict
        Mapping of results to memory
    """
    # Process the data
    filtered = table[table['bill_length_mm'] > 40]
    return {
        "filtered_table": filtered,
        "summary": f"Found {len(filtered)} records matching criteria"
    }

tool = FunctionTool(
    function=apply_proprietary_algorithm,
    requires=["table"],
    provides=["filtered_table", "summary"],
    purpose="Filters penguins by bill length using proprietary algorithm"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
ui.servable()
```

### Tool that wraps external APIs

Wrap external APIs or libraries as tools:

```python
import lumen.ai as lmai
from lumen.ai.tools import FunctionTool
import requests

def fetch_weather(location: str) -> str:
    """
    Fetch current weather for a location.
    
    Parameters
    ----------
    location : str
        City name
        
    Returns
    -------
    str
        Weather information
    """
    # In production, use a real weather API
    return f"Weather for {location}: Sunny, 72°F"

tool = FunctionTool(
    function=fetch_weather,
    purpose="Retrieves current weather information for any location"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
ui.servable()
```

### Complete example: Data validation tool

Here's a complete tool that validates data and stores results:

```python
import pandas as pd
import lumen.ai as lmai
from lumen.ai.tools import FunctionTool

def validate_data_quality(table: pd.DataFrame) -> dict:
    """
    Validates data quality and generates a report.
    
    Checks for missing values, duplicates, and data type consistency.
    
    Parameters
    ----------
    table : pd.DataFrame
        The data table to validate
        
    Returns
    -------
    dict
        Validation report with metrics
    """
    report = {
        "total_rows": len(table),
        "total_columns": len(table.columns),
        "missing_values": table.isnull().sum().to_dict(),
        "duplicate_rows": table.duplicated().sum(),
        "memory_usage_mb": table.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Check for issues
    issues = []
    if any(table.isnull().sum() > 0):
        issues.append("⚠️ Missing values detected")
    if report["duplicate_rows"] > 0:
        issues.append("⚠️ Duplicate rows found")
    
    report["issues"] = issues
    report["status"] = "✓ Data looks good" if not issues else "⚠️ Issues found"
    
    return report

tool = FunctionTool(
    function=validate_data_quality,
    requires=["table"],
    provides=["data_quality_report"],
    purpose="Validates data quality and generates a report of issues",
    formatter="Data quality validation complete: {output[status]}"
)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[tool]
)
ui.servable()
```

### Guidelines for custom tools

**Write clear docstrings.** The LLM uses your docstring to understand when and how to use the tool:

```python
def my_tool(data: list) -> str:
    """
    Meaningful description of what the tool does.
    
    Be specific about inputs and outputs.
    
    Parameters
    ----------
    data : list
        What the data represents
        
    Returns
    -------
    str
        What the return value contains
    """
```

**Use explicit type hints.** Type hints help the LLM provide correct arguments:

```python
def process_data(data: list[float], threshold: int) -> dict:
    """Process numeric data against a threshold."""
```

**Name parameters descriptively.** Avoid generic names like `x` or `val`:

```python
# ✓ Good
def calculate_average(numbers: list[float]) -> float:

# ✗ Avoid
def calculate(x: list[float]) -> float:
```

**Keep tools focused.** One tool should do one thing well:

```python
# ✓ Good - single responsibility
def validate_email(email: str) -> bool:

# ✗ Avoid - too many responsibilities
def validate_and_process_user_data(data: dict):
```

**Handle errors gracefully.** Return meaningful messages, not exceptions:

```python
def process_data(data: list) -> str:
    """Process data and return results."""
    if not data:
        return "No data provided - nothing to process"
    
    try:
        result = sum(data) / len(data)
        return f"Average: {result:.2f}"
    except Exception as e:
        return f"Error processing data: {e}"
```

**Test with actual LLM.** Different models call tools differently. Test with your configured LLM:

```python
# Test your tool works as expected
tool_instance = my_tool_function
result = tool_instance(test_data)
print(result)
```

**Document side effects.** If your tool modifies memory or has side effects, document it:

```python
def export_results(table, filename: str) -> str:
    """
    Export table to CSV file.
    
    Side effects: Creates a file on disk.
    """
```

### Combine multiple tools

Use multiple tools together to create powerful workflows:

```python
import lumen.ai as lmai
from lumen.ai.tools import FunctionTool, DocumentLookup

def get_aggregates(table) -> dict:
    """Get summary statistics."""
    return {
        "min": table['bill_length_mm'].min(),
        "max": table['bill_length_mm'].max(),
        "mean": table['bill_length_mm'].mean(),
    }

def apply_filter(table, species: str):
    """Filter by species."""
    return table[table['species'] == species]

ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[
        FunctionTool(get_aggregates, requires=["table"]),
        FunctionTool(apply_filter, requires=["table"]),
        DocumentLookup()
    ]
)
ui.servable()
```

Now agents can use all three tools in combination to answer complex questions about your data.
