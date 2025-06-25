# Configuring Models for Different Agents

Lumen AI uses different models for different types of agents and tasks to optimize performance and cost. Each agent can be configured to use a specific model by leveraging the `model_kwargs` parameter in your LLM configuration.

## How Agent Model Selection Works

### Automatic Model Selection

Agents automatically determine which model to use based on their `llm_spec_key` property, which converts the agent class name to a snake_case identifier:

- `SQLAgent` → `sql`
- `VegaLiteAgent` → `vega_lite`
- `AnalysisAgent` → `analysis`
- `ChatAgent` → `chat`
- `DbtslAgent` → `dbtsl`

If no specific model is configured for an agent, it falls back to the `default` model.

### Model Categories

Lumen AI organizes models into these categories:

- **`default`**: General-purpose model for most tasks
- **`sql`**: Optimized for SQL query generation and database operations
- **`vega_lite`**: Specialized for creating Vega-Lite visualizations
- **`reasoning`**: Advanced model for complex reasoning tasks
- **`ui`**: Used for user interface interactions and simple confirmations

## Default Model Configuration

Here are the default model assignments for OpenAI:

```python
model_kwargs = {
    "default": {"model": "gpt-4o-mini"},
    "sql": {"model": "gpt-4.1-mini"},
    "vega_lite": {"model": "gpt-4.1-mini"},
    "reasoning": {"model": "gpt-4.1-mini"},
}
```

## Customizing Models for Specific Agents

### Basic Configuration

You can override the default models for specific agents:

```python
import lumen.ai as lmai

# Configure different models for different agent types
config = {
    "default": {"model": "gpt-4o-mini"},        # General tasks
    "sql": {"model": "gpt-4o"},                 # SQL generation
    "vega_lite": {"model": "gpt-4o"},           # Visualization creation
    "reasoning": {"model": "o1-mini"},          # Complex reasoning
    "chat": {"model": "gpt-4o-mini"},           # Chat interactions
    "analysis": {"model": "gpt-4o"},            # Data analysis
}

llm = lmai.llm.OpenAI(model_kwargs=config)
ui = lmai.ui.ExplorerUI('<your-data-file>', llm=llm)
ui.servable()
```

### Advanced Model Configuration

You can also specify additional parameters for each model:

```python
import lumen.ai as lmai

config = {
    "default": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    },
    "sql": {
        "model": "gpt-4o",
        "temperature": 0.1,  # Lower temperature for more deterministic SQL
    },
    "vega_lite": {
        "model": "gpt-4o",
        "temperature": 0.2,  # Slightly higher for creative visualizations
    },
    "reasoning": {
        "model": "o1-mini",
        "temperature": 0.4,  # Higher temperature for complex reasoning
    },
}

llm = lmai.llm.OpenAI(model_kwargs=config)
```
:::

::::

## Agent-Specific Considerations

### SQL Agent
- **Purpose**: Generates and executes SQL queries
- **Recommended**: More capable models (e.g., gpt-4o, claude-3-5-sonnet)
- **Why**: SQL generation requires strong logical reasoning and syntax knowledge

### VegaLiteAgent
- **Purpose**: Creates Vega-Lite visualization specifications
- **Recommended**: Creative, capable models (e.g., gpt-4o, claude-3-5-sonnet)
- **Why**: Visualization requires understanding of design principles and JSON structure

### ChatAgent
- **Purpose**: General conversation and guidance
- **Recommended**: Cost-effective models (e.g., gpt-4o-mini, claude-3-5-haiku)
- **Why**: Simple interactions don't require the most advanced reasoning

### AnalysisAgent
- **Purpose**: Performs custom data analysis
- **Recommended**: Advanced models (e.g., gpt-4o, claude-3-5-sonnet)
- **Why**: Analysis requires deep understanding of statistical concepts

### DbtslAgent
- **Purpose**: Queries dbt Semantic Layer
- **Recommended**: SQL-capable models (e.g., gpt-4o, claude-3-5-sonnet)
- **Why**: Requires understanding of business metrics and semantic modeling
