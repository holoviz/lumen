# :material-robot: Agents

**Agents are specialized workers that answer different types of questions.**

SQLAgent writes queries. VegaLiteAgent creates charts. ChatAgent answers questions. Each agent has a specific job.

Most users never customize agents. The eight default agents handle typical data exploration needs.

**See also:** [Using Lumen AI](../getting_started/using_lumen_ai.md) — Guide to asking effective questions and exploring data.

## Skip to

- [See which agents exist](#default-agents) - What each agent does
- [Add custom agents](#add-a-custom-agent) - Extend Lumen with new capabilities  
- [Remove agents](#use-specific-agents-only) - Use only some agents
- [Configure agent models](#use-different-models-per-agent) - Control which LLM each agent uses
- [Customize agent instructions](#customizing-agent-instructions) - Override agent prompts and behavior

## Default agents

Lumen includes eight agents automatically. You don't need to configure anything.

| Agent | What it does |
|-------|-------------|
| **SQLAgent** | Writes and runs SQL queries |
| **VegaLiteAgent** | Creates charts and visualizations |
| **DeckGLAgent** | Creates 3D map visualizations for geographic data |
| **ChatAgent** | Answers questions and provides guidance |
| **TableListAgent** | Lists available tables and columns |
| **DocumentListAgent** | Manages uploaded documents |
| **ValidationAgent** | Checks if results answer the question |

These agents work together automatically. The coordinator picks which agents to use for each question.

## Use specific agents only

Include only the agents you need:

``` py title="Limit to specific agents"
import lumen.ai as lmai
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent, DeckGLAgent

ui = lmai.ExplorerUI(
    data='penguins.csv',
    default_agents=[ChatAgent, SQLAgent, VegaLiteAgent, DeckGLAgent]
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

``` py title="Minimal custom agent"
import lumen.ai as lmai
from lumen.ai.context import ContextModel
from pydantic import Field

class MyInputs(ContextModel):
    data: dict = Field(description="The data to process")

class MyOutputs(ContextModel):
    summary: str = Field(description="Summary result")

class SummaryAgent(lmai.agents.Agent):
    purpose = "Creates executive summaries of data"
    
    input_schema = MyInputs
    output_schema = MyOutputs
    
    async def respond(self, messages, context, **kwargs):
        # Your logic here
        return [outputs], context

ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[SummaryAgent()]  # (1)!
)
ui.servable()
```

1. Adds your agent alongside the default agents

See [Creating custom agents](#creating-custom-agents) below for complete examples.

## Use different models per agent

Configure which LLM model each agent uses:

``` py title="Different models per agent" hl_lines="4-7"
import lumen.ai as lmai

model_config = {
    "default": {"model": "gpt-4o-mini"},  # Cheap model for most agents
    "sql": {"model": "gpt-4o"},           # Powerful model for SQL
    "vega_lite": {"model": "gpt-4o"},     # Powerful model for charts
    "deck_gl": {"model": "gpt-4o"},       # Powerful model for 3D maps
    "chat": {"model": "gpt-4o"},          # Powerful model for analysis
}

llm = lmai.llm.OpenAI(model_kwargs=model_config)

ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

**Model types match agent names:**

- SQLAgent uses the `"sql"` model
- VegaLiteAgent uses the `"vega_lite"` model
- DeckGLAgent uses the `"deck_gl"` model
- ChatAgent uses the `"chat"` model (falls back to `"default"` if not specified)

Agent class names are converted to model keys automatically (e.g., `SQLAgent` → `"sql"`, `VegaLiteAgent` → `"vega_lite"`, `DeckGLAgent` → `"deck_gl"`).

See [LLM Providers](llm_providers.md) for complete details.

## Customizing agent instructions

You can customize the instructions or system prompts for any agent using `template_overrides`. This is useful for injecting domain knowledge or changing the agent's behavior for specific tasks.

### Overriding instructions via subclassing

The most common way to customize an agent's instructions is to subclass it and provide a `template_overrides` dictionary:

``` py title="Customized SQL Agent"
from lumen.ai import ExplorerUI
from lumen.ai.agents.sql import SQLAgent

INSTRUCTION_OVERRIDE = """
{{ super() }}

When querying the database, always prioritize the `current_year` filter 
unless the user specifically asks for historical data.
"""

class UXSQLAgent(SQLAgent):

    template_overrides = {
        "main": {"instructions": INSTRUCTION_OVERRIDE}
    }

ui = ExplorerUI(agents=[UXSQLAgent()])
ui.servable()
```

### Global and class-level overrides

You can also apply overrides globally to all agents or to specific agent classes directly. This is useful for adding context that applies across the entire application without needing to subclass.

``` py title="Global and class-level overrides"
import lumen.ai as lmai

# Add context to ALL agents (planning phase)
lmai.actor.Actor.template_overrides = {
    "main": {
        "global": "{{ super() }}\nThis application focuses on weather data analysis."
    }
}

# Add context to ChatAgent specifically
lmai.agents.ChatAgent.template_overrides = {
    "main": {
        "instructions": "{{ super() }}\nYou are a professional meteorologist."
    }
}
```

**Key concepts:**

*   **`{{ super() }}`**: Always include this Jinja2 tag to preserve the default instructions or context.
*   **`main`**: The primary prompt group for the agent.
*   **`instructions`**: The system instructions specifically for that agent.
*   **`global`**: Shared context injected into the system prompt for all agents.

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
- You can [use tools instead](tools.md) (tools don't require async/await)
- A built-in agent already handles it

### Basic custom agent structure

``` py title="Custom agent structure" hl_lines="14-16"
import lumen.ai as lmai
from lumen.ai.context import ContextModel
from pydantic import Field

# Define what the agent needs
class MyInputs(ContextModel):
    pipeline: object = Field(description="Data pipeline to process")

# Define what the agent provides
class MyOutputs(ContextModel):
    summary: str = Field(description="Summary of findings")

class MyAgent(lmai.agents.Agent):
    purpose = "Summarizes data in executive format"
    
    input_schema = MyInputs  # (1)!
    output_schema = MyOutputs  # (2)!
    
    prompts = {
        "main": {
            "template": "Summarize this data: {{ memory['data'] }}"
        }
    }
    
    async def respond(self, messages, context, **kwargs):
        # Render prompt
        system = await self._render_prompt("main", messages, context)
        
        # Get LLM response
        response = await self.llm.invoke(messages, system=system)
        
        # Return outputs and updated context
        return [response], {"summary": str(response)}
```

1. Agent requires `pipeline` in context to run
2. Agent adds `summary` to context after running

### Complete working example

This agent calculates statistical metrics:

``` py title="Statistics agent" linenums="1"
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
        interpretation = await self.llm.invoke(messages, system=system)
        
        # Return results
        return [interpretation], {"statistics": str(interpretation)}

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

**`input_schema`** - TypedDict defining what data the agent needs from [context](context.md). The agent can only run when these requirements are met.

**`output_schema`** - TypedDict defining what data the agent adds to [context](context.md). Other agents can use these outputs.

**`prompts`** - Dictionary of prompt templates. Most agents only need a "main" prompt. [See Prompts guide](prompts.md) for customization options.

**`respond()`** - The async method that does the work. Must return `(outputs_list, updated_context_dict)`.

### Control when agents are used

Use `conditions` to specify when the agent should run:

``` py title="Agent with conditions" hl_lines="5-9"
import param

class ReportAgent(lmai.agents.Agent):
    purpose = "Creates PDF reports"
    
    conditions = param.List(default=[
        "Use when user explicitly asks for a report or PDF",
        "Use after data analysis is complete",
        "NOT for simple questions or queries"
    ])
    
    input_schema = MyInputs
    output_schema = MyOutputs
```

The coordinator reads these conditions when deciding which agent to use.

### Prevent agent conflicts

Use `not_with` to prevent agents from being used together:

``` py title="Prevent conflicting agents" hl_lines="5"
class FastSummaryAgent(lmai.agents.Agent):
    purpose = "Quick data summaries"
    
    not_with = param.List(default=["DetailedAnalysisAgent"])
```

### Common patterns

=== "API Integration"

    ``` py title="Call external APIs"
    import httpx

    class WeatherAgent(lmai.agents.Agent):
        purpose = "Fetches current weather data"
        
        async def respond(self, messages, context, **kwargs):
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.weather.gov/...")
                weather_data = response.json()
            
            summary = f"Current temperature: {weather_data['temp']}°F"
            return [summary], {"weather": summary}
    ```

=== "File Processing"

    ``` py title="Extract PDF text"
    class PDFAgent(lmai.agents.Agent):
        purpose = "Extracts text from PDF documents"
        
        async def respond(self, messages, context, **kwargs):
            documents = context.get('documents', [])
            
            extracted_text = []
            for doc in documents:
                if doc['type'] == 'pdf':
                    text = extract_pdf_text(doc['content'])
                    extracted_text.append(text)
            
            return [extracted_text], {"pdf_text": extracted_text}
    ```

=== "External Library"

    ``` py title="Data quality checks"
    import great_expectations as gx

    class DataQualityAgent(lmai.agents.Agent):
        purpose = "Checks data quality using Great Expectations"
        
        async def respond(self, messages, context, **kwargs):
            df = context['pipeline'].data
            
            # Run validations
            results = run_quality_checks(df)
            
            # Summarize findings
            system = await self._render_prompt(
                "main", messages, context, results=results
            )
            summary = await self.llm.invoke(messages, system=system)
            
            return [summary], {"quality_report": str(summary)}
    ```

## Common issues

### "Agent has unmet requirements"

The agent's `input_schema` requires data that doesn't exist in context.

**How to fix:**

``` py title="Make fields optional"
from typing import NotRequired

class MyInputs(ContextModel):
    pipeline: object  # Required
    analysis: NotRequired[str]  # Optional
```

Or ensure another agent provides the required data first.

### Agent never gets invoked

The coordinator doesn't think the agent is relevant.

**How to fix:**

1. Make the `purpose` more specific and clear
2. Add `conditions` that describe when to use it
3. Check that `input_schema` requirements can be satisfied
4. Enable `log_level='DEBUG'` in the UI to see coordinator decisions

### Agent fails with "KeyError"

The agent tried to access context data that doesn't exist.

!!! warning "Always check before accessing context"

    ``` py hl_lines="2 5-7"
    # Bad - assumes 'data' exists
    data = context['data']  # ❌ KeyError if missing

    # Good - checks first
    data = context.get('data')  # ✅ Returns None if missing
    if data is None:
        return [{"error": "No data available"}], context
    ```

## Best practices

**Keep agents focused.** One agent should do one thing well. Don't create a "do everything" agent.

**Write clear purposes.** The coordinator uses `purpose` to decide when to invoke agents. Make it specific and actionable.

**Test with real queries.** Different LLM models behave differently. Test your agent with your actual LLM.

**Handle missing data gracefully.** Always check for required data before using it. Provide helpful error messages.

**Use tools for simple functions.** If your agent doesn't need async/await or complex prompting, [use a tool instead](tools.md).

**Don't duplicate built-ins.** Check if a built-in agent already does what you need before creating a custom one.
