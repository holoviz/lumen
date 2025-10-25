# Agents

Agents are specialized task performers in Lumen AI. Each agent handles a specific role—from querying data to creating visualizations to analyzing results. The system comes with built-in agents you can use immediately, and you can create custom agents for specialized workflows.

## Built-in agents

Lumen AI includes eight default agents that handle common data exploration tasks:

**TableListAgent**

- Lists available tables in your datasets
- Responds to questions about data structure and schema
- Helps users understand what data is available

**ChatAgent**

- Engages in general conversation and technical discussion
- Answers questions about programming and APIs
- Provides information about data without querying

**DocumentListAgent**

- Manages and lists uploaded documents
- Provides context about document sources
- Helps users navigate document collections

**AnalystAgent**

- Analyzes query results and explains findings
- Generates insights from data
- Breaks down complex results into understandable points

**SourceAgent**

- Handles data source uploads and connections
- Manages multiple datasets
- Allows adding or replacing data sources

**SQLAgent**

- Generates SQL queries from natural language questions
- Executes queries against databases
- Ensures valid, optimized SQL

**VegaLiteAgent**

- Creates interactive visualizations (charts, plots, maps)
- Generates Vega-Lite specifications from queries
- Enables visual data exploration

**ValidationAgent**

- Validates query results for correctness
- Checks for data anomalies
- Catches potential errors before presenting results

## Using agents

### Use default agents

By default, all eight agents are included automatically. Just create an ExplorerUI instance:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(data='penguins.csv')
ui.servable()
```

The coordinator will route queries to the appropriate agent based on what the user asks.

### Add custom agents

Pass additional agents using the `agents` parameter:

```python
import lumen.ai as lmai

class CustomSummarizerAgent(lmai.agents.Agent):
    purpose = "Summarizes query results"
    # ... agent implementation ...

ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[CustomSummarizerAgent()]
)
ui.servable()
```

Your custom agents work alongside the default agents. All agents are available to the coordinator.

### Replace default agents

Override which agents are included using `default_agents`:

```python
import lumen.ai as lmai
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent

ui = lmai.ExplorerUI(
    data='penguins.csv',
    default_agents=[ChatAgent, SQLAgent, VegaLiteAgent]
)
ui.servable()
```

Only the agents you specify are used. TableListAgent, AnalystAgent, and others are not included.

### Combine custom and default agents

Use both together by setting `default_agents` AND `agents`:

```python
import lumen.ai as lmai
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent

class CustomReportAgent(lmai.agents.Agent):
    purpose = "Generates formatted reports"
    # ... agent implementation ...

ui = lmai.ExplorerUI(
    data='penguins.csv',
    default_agents=[ChatAgent, SQLAgent, VegaLiteAgent],
    agents=[CustomReportAgent()]
)
ui.servable()
```

The UI will have the three default agents plus your custom agent.

### Disable specific agents

To use all defaults except one, filter them:

```python
import lumen.ai as lmai
from lumen.ai.agents import (
    TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent,
    SourceAgent, SQLAgent, VegaLiteAgent
)

# All defaults except ValidationAgent
ui = lmai.ExplorerUI(
    data='penguins.csv',
    default_agents=[
        TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent,
        SourceAgent, SQLAgent, VegaLiteAgent
    ]
)
ui.servable()
```

### Customize agent models

Different agents can use different LLM models. Configure which model each agent uses:

```python
import lumen.ai as lmai

model_config = {
    "default": {"model": "gpt-4o-mini"},
    "sql": {"model": "gpt-4o"},          # SQLAgent uses gpt-4o
    "vega_lite": {"model": "gpt-4o"},    # VegaLiteAgent uses gpt-4o
    "chat": {"model": "gpt-4o-mini"},    # ChatAgent uses cheaper model
}

llm = lmai.llm.OpenAI(model_kwargs=model_config)

ui = lmai.ExplorerUI(
    data='penguins.csv',
    llm=llm
)
ui.servable()
```

Agents automatically select the right model based on their task. See [LLM Providers](llm_providers.md) for full configuration details.

### Analyze available agents

Check which agents are included in your UI:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(data='penguins.csv')

for agent in ui._coordinator.agents:
    print(f"Agent: {agent.name}")
    print(f"Purpose: {agent.purpose}")
    print(f"Requires: {agent.requires}")
    print(f"Provides: {agent.provides}")
    print()
```

## Creating custom agents

Once you're comfortable with the built-in agents, you can create custom agents for specialized tasks.

### What are custom agents?

Custom agents extend Lumen AI's capabilities by handling domain-specific tasks. They follow the same structure as built-in agents but implement your own logic.

Each agent declares what context it needs (`requires`), what it provides (`provides`), and when it should be used (`purpose`, `conditions`).

### Core concepts

**`requires`**

List of context values the agent needs from memory to operate. For example, `requires=["source"]` means the agent requires a data source.

**`provides`**

List of context values the agent adds or updates in memory. For example, `provides=["summary"]` means the agent creates a summary.

**`purpose`**

A clear description of what the agent does. The coordinator uses this to decide when to invoke the agent.

**`conditions`**

Specific criteria that determine when the agent should be invoked. While `purpose` describes what the agent does, conditions specify the precise situations requiring the agent.

### Agent structure

All agents inherit from the `Agent` base class:

```python
import lumen.ai as lmai
from lumen.ai.llm import Message

class CustomAgent(lmai.agents.Agent):
    
    purpose = "Describes what this agent does"
    
    requires = ["context_key"]     # What memory this needs
    provides = ["output_key"]      # What memory this creates
    
    prompts = {
        "main": {
            "template": "Your prompt template here",
        }
    }
    
    async def respond(self, messages: list[Message], **kwargs):
        # Implement agent logic here
        pass
```

### Build your first custom agent

**Step 1: Define purpose and responsibilities**

Describe what the agent does:

```python
purpose = "Summarizes documents and generates concise summaries"
```

**Step 2: Declare context**

Specify what the agent needs and what it creates:

```python
requires = ["documents"]   # Agent needs documents in memory
provides = ["summary"]      # Agent will create a summary
```

**Step 3: Write the prompt**

Create a template to guide the LLM:

```python
prompts = {
    "main": {
        "template": "Summarize these documents:\n{document_texts}"
    }
}
```

**Step 4: Implement respond**

Fetch data, render the prompt, call the LLM:

```python
async def respond(self, messages: list[Message], **kwargs):
    # Get data from memory
    documents = self._memory["documents"]
    
    # Prepare data for prompt
    document_texts = "\n".join(documents)
    
    # Render prompt with context
    system = await self._render_prompt("main", messages, document_texts=document_texts)
    
    # Stream response from LLM
    summary = await self._stream(messages, system)
    
    # Store result in memory
    self._memory["summary"] = summary
    
    return summary
```

**Step 5: Register the agent**

Add your agent to ExplorerUI:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[CustomSummarizerAgent()]
)
ui.servable()
```

### Example: Document Summarizer

Here's a complete custom agent:

```python
import lumen.ai as lmai
from lumen.ai.llm import Message

class DocumentSummarizerAgent(lmai.agents.Agent):
    """Summarizes uploaded documents into concise overviews."""
    
    purpose = "Reads documents and generates concise, actionable summaries"
    
    requires = ["documents"]
    provides = ["summary"]
    
    prompts = {
        "main": {
            "template": """Analyze and summarize these documents in 3-4 bullet points:

{document_texts}

Focus on key insights and actionable information."""
        }
    }
    
    async def respond(self, messages: list[Message], **kwargs):
        documents = self._memory.get("documents", [])
        
        if not documents:
            return "No documents available to summarize."
        
        # Combine document contents
        document_texts = ""
        for doc in documents:
            content = doc.get("content", "")
            document_texts += f"--- {doc.get('name', 'Document')} ---\n{content}\n\n"
        
        # Render and invoke LLM
        system = await self._render_prompt(
            "main", 
            messages, 
            document_texts=document_texts
        )
        summary = await self._stream(messages, system)
        
        # Store in memory
        self._memory["summary"] = summary
        
        return summary
```

### Control when agents are invoked

Use `conditions` to specify precise situations:

```python
class AnalysisAgent(lmai.agents.Agent):
    
    purpose = "Analyzes data query results"
    
    conditions = [
        "Use only after SQL queries have been executed",
        "Use when user asks for interpretation of results",
        "NOT for data exploration queries"
    ]
    
    requires = ["sql_results", "source"]
    provides = ["analysis"]
```

### Prevent conflicts

Use `not_with` to prevent agents from being invoked together:

```python
class AIAgent(lmai.agents.Agent):
    
    purpose = "Provides AI-generated recommendations"
    
    not_with = ["UserApprovalAgent"]
```

### Use different models for different tasks

Specify which LLM model to use:

```python
prompts = {
    "main": {
        "template": "Your template",
        "llm_spec": "reasoning"  # Uses reasoning model from model_kwargs
    }
}
```

This lets you use cheaper models for simple tasks and more capable models for complex reasoning.

### Access memory

Agents read and write to global memory:

```python
async def respond(self, messages: list[Message], **kwargs):
    # Read from memory
    source = self._memory.get("source")
    table = self._memory.get("table")
    
    # Write to memory
    self._memory["my_output"] = result
    
    # Check if key exists
    if "sql_results" in self._memory:
        results = self._memory["sql_results"]
```

### Stream responses

Show progress for long operations:

```python
async def respond(self, messages: list[Message], **kwargs):
    result = await self._stream(messages, system_prompt)
    return result
```

### Best practices

**Start simple.** Begin with a single focused task. Expand complexity later.

**Write clear purpose statements.** Help the coordinator understand when to invoke the agent.

**Declare requirements explicitly.** List all memory keys your agent needs. Incomplete declarations cause silent failures.

**Use prompts effectively.** Keep prompts concise and structured. Use templates for consistency.

**Test with your LLM.** Different models behave differently. Test with the same model configuration as production.

**Store results in memory.** Always save outputs so other agents can access them.

**Handle missing context gracefully.** Check for required memory keys and provide helpful messages if they're missing.

**Avoid circular dependencies.** Don't create agents that require outputs from agents requiring this agent's outputs.
