# Agents

**Agents are specialized workers that answer different types of questions.**

SQLAgent writes queries. VegaLiteAgent creates charts. ChatAgent answers questions. Each agent has a specific job.

Most users never customize agents. The eight default agents handle typical data exploration needs.

## Skip to

- [See which agents exist](#default-agents) - What each agent does
- [Add custom agents](#add-a-custom-agent) - Extend Lumen with new capabilities  
- [Remove agents](#use-specific-agents-only) - Use only some agents
- [Configure agent models](#use-different-models-per-agent) - Control which LLM each agent uses

## Default agents

Lumen includes eight agents automatically. You don't need to configure anything.

| Agent | What it does |
|-------|-------------|
| **SQLAgent** | Writes and runs SQL queries |
| **VegaLiteAgent** | Creates charts and visualizations |
| **AnalystAgent** | Explains query results and finds insights |
| **ChatAgent** | Answers questions and provides guidance |
| **TableListAgent** | Lists available tables and columns |
| **DocumentListAgent** | Manages uploaded documents |
| **SourceAgent** | Handles data uploads |
| **ValidationAgent** | Checks if results answer the question |

These agents work together automatically. The coordinator picks which agents to use for each question.

## Use specific agents only

Include only the agents you need:

```python
import lumen.ai as lmai
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent

ui = lmai.ExplorerUI(
    data='penguins.csv',
    default_agents=[ChatAgent, SQLAgent, VegaLiteAgent]
)
ui.servable()
```

**Why limit agents?**

- Faster planning (fewer options to consider)
- Lower costs (fewer agents = fewer LLM calls during planning)
- Simpler behavior (predictable agent selection)

Most users should keep all default agents. Only customize if you have specific needs.

## Add a custom agent

Add your own agent for specialized tasks:

```python
import lumen.ai as lmai

class SummaryAgent(lmai.agents.Agent):
    purpose = "Creates executive summaries of data"
    
    # Define what this agent needs to run
    input_schema = ...  # Pydantic model
    output_schema = ... # Pydantic model
    
    async def respond(self, messages, context, **kwargs):
        # Your logic here
        return outputs, context

ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[SummaryAgent()]  # Adds to default agents
)
ui.servable()
```

See [Creating custom agents](#creating-custom-agents) below for complete examples.

## Use different models per agent

Configure which LLM model each agent uses:

```python
import lumen.ai as lmai

model_config = {
    "default": {"model": "gpt-4o-mini"},  # Cheap model for most agents
    "sql": {"model": "gpt-4o"},           # Powerful model for SQL
    "vega_lite": {"model": "gpt-4o"},     # Powerful model for charts
    "analyst": {"model": "gpt-4o"},       # Powerful model for analysis
}

llm = lmai.llm.OpenAI(model_kwargs=model_config)

ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

**Model types match agent names:**

- SQLAgent uses the `"sql"` model
- VegaLiteAgent uses the `"vega_lite"` model  
- ChatAgent uses the `"chat"` model
- AnalystAgent uses the `"analyst"` model

See [LLM Providers](llm_providers.md#model-types) for complete details.

## Creating custom agents

Custom agents let you add specialized capabilities to Lumen.

### When to create a custom agent

Create a custom agent when:

- You need domain-specific analysis (financial metrics, scientific calculations)
- You want to integrate external APIs or services
- You need specialized data transformations
- Built-in agents don't match your workflow

Don't create a custom agent when:

- You can solve it with custom analyses (simpler approach)
- You can use tools instead (tools don't require async/await)
- A built-in agent already handles it

### Basic custom agent structure

```python
import lumen.ai as lmai
from lumen.ai.context import ContextModel
from pydantic import Field

# Define what the agent needs
class MyInputs(ContextModel):
    data: dict = Field(description="The data to process")

# Define what the agent provides
class MyOutputs(ContextModel):
    summary: str = Field(description="Summary of findings")

class MyAgent(lmai.agents.Agent):
    purpose = "Summarizes data in executive format"
    
    input_schema = MyInputs
    output_schema = MyOutputs
    
    prompts = {
        "main": {
            "template": "Summarize this data: {{ context.data }}"
        }
    }
    
    async def respond(self, messages, context, **kwargs):
        # Render prompt
        system = await self._render_prompt("main", messages, context)
        
        # Get LLM response
        summary = await self._stream(messages, system)
        
        # Return outputs
        return {"summary": summary}, context
```

### Complete working example

This agent calculates statistical metrics:

```python
import lumen.ai as lmai
from lumen.ai.context import ContextModel
from pydantic import Field
import pandas as pd

class StatsInputs(ContextModel):
    pipeline: object = Field(description="Data pipeline")

class StatsOutputs(ContextModel):
    statistics: str = Field(description="Statistical summary")

class StatisticsAgent(lmai.agents.Agent):
    purpose = "Calculates descriptive statistics for numerical columns"
    
    input_schema = StatsInputs
    output_schema = StatsOutputs
    
    prompts = {
        "main": {
            "template": """
Analyze these statistics and explain key findings:

{{ stats }}

Focus on:

- Notable values (very high/low)
- Spread and variability  
- Potential outliers
"""
        }
    }
    
    async def respond(self, messages, context, **kwargs):
        # Get data
        pipeline = context['pipeline']
        df = pipeline.data
        
        # Calculate stats
        stats = df.describe().to_string()
        
        # Get LLM interpretation
        system = await self._render_prompt("main", messages, context, stats=stats)
        interpretation = await self._stream(messages, system)
        
        # Return results
        return {"statistics": interpretation}, context

# Use the agent
ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[StatisticsAgent()]
)
ui.servable()
```

Now you can ask "What are the statistics for this dataset?" and the agent will run.

### Agent components explained

**`purpose`** - One-sentence description of what the agent does. The coordinator uses this to decide when to invoke the agent.

**`input_schema`** - Pydantic model defining what data the agent needs from memory. The agent can only run when these requirements are met.

**`output_schema`** - Pydantic model defining what data the agent adds to memory. Other agents can use these outputs.

**`prompts`** - Dictionary of prompt templates. Most agents only need a "main" prompt.

**`respond()`** - The async method that does the work. Must return `(outputs_dict, updated_context)`.

### Control when agents are used

Use `conditions` to specify when the agent should run:

```python
class ReportAgent(lmai.agents.Agent):
    purpose = "Creates PDF reports"
    
    conditions = [
        "Use when user explicitly asks for a report or PDF",
        "Use after data analysis is complete",
        "NOT for simple questions or queries"
    ]
    
    input_schema = MyInputs
    output_schema = MyOutputs
```

The coordinator reads these conditions when deciding which agent to use.

### Prevent agent conflicts

Use `not_with` to prevent agents from being used together:

```python
class FastSummaryAgent(lmai.agents.Agent):
    purpose = "Quick data summaries"
    
    not_with = ["DetailedAnalysisAgent"]  # Don't use both in same plan
```

### Common patterns

**Agent that calls an API:**

```python
class WeatherAgent(lmai.agents.Agent):
    purpose = "Fetches current weather data"
    
    async def respond(self, messages, context, **kwargs):
        # Call external API
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.weather.gov/...")
            weather_data = response.json()
        
        # Format for user
        summary = f"Current temperature: {weather_data['temp']}°F"
        
        return {"weather": summary}, context
```

**Agent that processes files:**

```python
class PDFAgent(lmai.agents.Agent):
    purpose = "Extracts text from PDF documents"
    
    async def respond(self, messages, context, **kwargs):
        documents = context['documents']
        
        extracted_text = []
        for doc in documents:
            if doc['type'] == 'pdf':
                # Extract text from PDF
                text = extract_pdf_text(doc['content'])
                extracted_text.append(text)
        
        return {"pdf_text": extracted_text}, context
```

**Agent that uses external tools:**

```python
class DataQualityAgent(lmai.agents.Agent):
    purpose = "Checks data quality using Great Expectations"
    
    async def respond(self, messages, context, **kwargs):
        import great_expectations as gx
        
        df = context['pipeline'].data
        
        # Run validations
        results = run_quality_checks(df)
        
        # Summarize findings
        system = await self._render_prompt("main", messages, context, results=results)
        summary = await self._stream(messages, system)
        
        return {"quality_report": summary}, context
```

## Common issues

### "Agent has unmet requirements"

**What happened:** The agent's `input_schema` requires data that doesn't exist in memory.

**How to fix:**

1. Check what the agent needs in its `input_schema`
2. Make sure another agent provides that data first
3. Or adjust the `input_schema` to not require it

### Agent never gets invoked

**What happened:** The coordinator doesn't think the agent is relevant.

**How to fix:**

1. Make the `purpose` more specific and clear
2. Add `conditions` that describe when to use it
3. Check that `input_schema` requirements can be satisfied
4. Enable `verbose=True` to see why it wasn't selected

### Agent fails with "KeyError"

**What happened:** The agent tried to access memory that doesn't exist.

**How to fix:**

```python
# Bad - assumes 'data' exists
data = context['data']

# Good - checks first
data = context.get('data')
if data is None:
    return {"error": "No data available"}, context
```

Always use `.get()` for optional keys in context.

## Best practices

**Keep agents focused.** One agent should do one thing well. Don't create a "do everything" agent.

**Write clear purposes.** The coordinator uses `purpose` to decide when to invoke agents. Make it specific and actionable.

**Test with real queries.** Different LLM models behave differently. Test your agent with your actual LLM.

**Handle missing data gracefully.** Always check for required data before using it. Provide helpful error messages.

**Use tools for simple functions.** If your agent doesn't need async/await or complex prompting, use a tool instead.

**Don't duplicate built-ins.** Check if a built-in agent already does what you need before creating a custom one.
