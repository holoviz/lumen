# Custom Tools

Tools extend the capabilities of LLMs/agents by enabling them to access external data, perform specialized tasks, and interact dynamically with the environment.

While there are built-in tools available, users often need to create custom tools to perform domain-specific actions. For example, you might want to create a tool that performs a custom database query, queries an API, or runs specialized computations.

This guide covers how to:

- Create a custom tool using a simple function.
- Wrap the function with a `FunctionTool` for integration into your system.
- Use `requires` and `provides` parameters to manage context data.
- Provide comprehensive docstrings and type hints so LLMs can properly use the tool.

## Python Function as a Tool

1. **Write the function:**
   Implement the Python function that performs the desired action. The function can be as simple or complex as necessary.

2. **Add type hints and docstrings:**
   Provide type hints for all arguments and a meaningful docstring. The docstring should explain what the function does, what inputs it expects, and what outputs it returns. The LLM uses this information to call the tool correctly.

3. **Add the tool to the tool list:**
   Once wrapped, add your custom tool to the `tools` list so it becomes available to the system’s `Coordinator` and subsequent `Agent`s.

```python
def duckduckgo_search(queries: list[str]) -> dict:
    """
    Perform a DuckDuckGo search for the provided queries.

    Parameters
    ----------
    queries : list[str]
        Search queries.

    Returns
    -------
    dict
        A dictionary mapping each query to a list of search results.
        Each search result is a dict containing 'title' and 'url'.
    """
    import requests
    from bs4 import BeautifulSoup

    results = {}
    for query in queries:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", {"class": "result__a"}, href=True)

        results[query] = [
            {"title": link.get_text(strip=True), "url": link["href"]}
            for link in links
        ]
    return results

tools = [duckduckgo_search]  # This makes your function directly available as a tool
ui = lmai.ExplorerUI(tools=tools)
```

By sharing the `duckduckgo_search` tool with the `Coordinator`, `Agent`s can leverage this tool to provide more accurate and relevant information.

## `FunctionTool` for Controlled Integration

In more complex scenarios, you might want to use `FunctionTool` to help manage when the LLM is able to call your function. This tool provides additional context to the LLM, such as required inputs and outputs, and helps manage the flow of data between tools.

When tools are integrated into a system with memory, you may have data already loaded (e.g., a `table`, a `source` URL, or a `sql` query string) that your function needs. Conversely, you may want your tool’s output to be stored back into memory so other tools or subsequent requests can access it.

**`requires`**:
A list of memory keys that the tool expects to already exist. These values are automatically provided as arguments to the function and do not need to be requested from the LLM. For example, if your tool needs a `table` object that was previously retrieved from a database, you can specify `requires=["table"]`.

**`provides`**:
A list of memory keys that the tool will populate after it runs. For instance, if your function returns a new `sql` query string, you can specify `provides=["sql"]`.

**`formatter`**:
A string that formats the return value of the function for inclusion in the global context. The formatter accepts the `function`, `arguments`, and `output` as formatting variables. By default, the formatter is set to `"{function}({arguments}) returned: {output}"`.

Some useful keys you might use in `requires` and `provides` include:

- `table`: A DataFrame or similar structure representing tabular data.
- `source`: A string indicating a data source location (e.g., file path or URL).
- `sql`: A string containing an SQL query or command.

```python
from lmai.tools import FunctionTool

def apply_propietary_algorithm(table) -> str:
    """
    Shares the proprietary algorithm on the provided data,
    which SQLAgent can then use to query the data.

    Parameters
    ----------
    table : pd.DataFrame
        The input data table.
    """
    if table == "expected_table":
        memory["sql"] = sql = f"SELECT * FROM expected_table WHERE column > 10"
        return f"Please run this SQL expression: {sql}"
    else:
        return "Please continue on; this tool only works with the expected_table."

tool = FunctionTool(
    function=apply_propietary_algorithm,
    requires=["table"],  # require a table to be in memory
    provides=["sql"],  # provide the resulting SQL query
    purpose="Filter the table to only include rows where column > 10; only use if the table is 'expected_table'."
)

ui = lmai.ExplorerUI(tools=[tool])
```

In this example:

- The `run_propietary_algorithm` function requires a `table` to be in memory.
- If the table is the expected one, the function generates an SQL query and stores it back in memory.
- The `FunctionTool` is created with the `requires` and `provides` parameters to manage the flow of data.
- The `purpose` parameter provides a description of when and how to use the tool.
- The tool is added to the `tools` list and made available to the LLM.
- The LLM can now use this tool to filter tables based on the proprietary algorithm.

## Wrapping Existing Tools

You can wrap existing tools from libraries like LangChain and add them to your `tools` list by simply creating a `FunctionTool` that calls the tool's `invoke` method, or equivalent, and providing the `purpose` to guide the LLM on when to use the tool:

```python
from langchain_community.tools import DuckDuckGoSearchRun
from lmai.tools import FunctionTool

search = DuckDuckGoSearchRun()
tool = FunctionTool(function=search.invoke, purpose=search.description)
ui = lmai.ExplorerUI(tools=[tool])
```

This allows you to leverage the rich ecosystem of community and open-source tools without reinventing the wheel.

## Additional Best Practices

- **Use descriptive docstrings:**
  The more descriptive and clear your docstring, the easier it is for the LLM to understand when and how to apply your tool.

- **Provide explicit type hints:**
  Use Python type hints so the underlying Pydantic model can validate inputs. This ensures the LLM provides correctly formatted arguments.

- **Iteratively refine:**
  Start simple. Create your tool and test it. Then gradually add complexity (like `requires` and `provides`) as needed.
