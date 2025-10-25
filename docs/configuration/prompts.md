# Prompts

Customize how agents interact with LLMs by modifying prompts. Agents use Jinja2 templates organized into modular blocks that you can override or replace entirely.

## Understanding prompt structure

All prompts are built from a base template with modular blocks:

- `global` — Global instructions (rarely used)
- `datetime` — Current date and time
- `instructions` — Task-specific instructions
- `context` — Contextual information from memory
- `tools` — Available tools and their descriptions
- `examples` — Examples of expected behavior
- `errors` — Error messages and recovery guidance

The base template flows through these blocks to create the final system prompt sent to the LLM. Each agent can override specific blocks or replace the entire template.

### Template variables and context

Templates have access to variables from memory. Common variables include:

- `current_datetime` — Current date and time
- `memory` — Global memory dictionary
- `actor_name` — Name of the current agent
- `source_table_sep` — Table separator used in data sources
- `last_output` — Previous output (if this is a retry)
- `errors` — Error messages to address

Access these in your templates:

```python
template = """
Current time: {{ current_datetime.strftime('%Y-%m-%d') }}
Actor: {{ actor_name }}

Available data: {{ memory.get('visible_slugs', []) | join(', ') }}

{{ memory['data'] }}
"""

prompts = {
    "main": {
        "template": template
    }
}

agent = lmai.agents.ChatAgent(prompts=prompts)
```

### Template inheritance

Prompts use Jinja2 template inheritance. Custom templates can extend base templates:

```jinja2
{% extends 'Agent/main.jinja2' %}

{% block instructions %}
Your custom instructions here.
{% endblock %}

{% block context %}
Your custom context here.
{% endblock %}
```

This extends the base Agent template but overrides only the `instructions` and `context` blocks. Other blocks remain unchanged.

## Customizing prompts

### Override specific blocks

The simplest way to customize a prompt is using `template_overrides`. This lets you override specific blocks without rebuilding the entire template.

**Override instructions:**

```python
import lumen.ai as lmai

template_overrides = {
    "main": {
        "instructions": "Act as a weather expert and explain data using meteorological terms."
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

**Override context:**

```python
template_overrides = {
    "main": {
        "context": """
Available datasets:
{datasets}

Focus on the most important patterns in the data.
        """
    }
}

agent = lmai.agents.AnalystAgent(template_overrides=template_overrides)
```

**Override examples:**

```python
template_overrides = {
    "main": {
        "examples": """
Example 1: When asked about sales, respond with: "Sales increased by X% in Q2."
Example 2: Use bullet points for summaries, not paragraphs.
        """
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
```

**Override multiple blocks:**

```python
template_overrides = {
    "main": {
        "instructions": "Speak concisely and focus on key metrics.",
        "context": "Current data: {latest_data}",
        "examples": "- Keep responses under 100 words.\n- Use metrics, not descriptions."
    }
}

agent = lmai.agents.AnalystAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

### Extend with super()

Use `{{ super() }}` to extend the original instructions instead of replacing them completely:

```python
template_overrides = {
    "main": {
        "instructions": "{{ super() }} Additionally, respond in pirate speak: yarr, arr, ahoy!"
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

This keeps the original instructions and appends your custom text.

### Replace entire templates

For complete control, replace the template entirely:

```python
template = """
You are a helpful data assistant.

Available tables: {available_tables}

Schema information:
{schema}

User question: {question}

Provide a concise response.
"""

prompts = {
    "main": {
        "template": template
    }
}

agent = lmai.agents.ChatAgent(prompts=prompts)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

Or use a template file:

```python
prompts = {
    "main": {
        "template": "/path/to/my_prompt.jinja2"
    }
}

agent = lmai.agents.ChatAgent(prompts=prompts)
```

### Customize response models

Some agents use Pydantic models to structure LLM responses. Customize these models to change how the LLM formats its response:

```python
from pydantic import BaseModel, Field
from lumen.ai.models import Sql

class CustomSql(Sql):
    chain_of_thought: str = Field(
        description="Think step-by-step like a DuckDB expert."
    )

prompts = {
    "main": {
        "model": CustomSql
    }
}

agent = lmai.agents.SQLAgent(prompts=prompts)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

The field names must match the original model. Only override fields you want to change.

### Multiple prompts per agent

Some agents have multiple prompts for different tasks. For example, SQLAgent has:

- `main` — Generate SQL query
- `select_discoveries` — Select relevant discoveries for the query
- `check_sufficiency` — Validate query results
- `retry_output` — Fix errors in previous output

Customize specific prompts:

```python
template_overrides = {
    "main": {
        "instructions": "Generate optimized SQL queries."
    },
    "select_discoveries": {
        "instructions": "Select the most relevant discoveries for this query."
    },
    "check_sufficiency": {
        "instructions": "Validate that the query results are sufficient and complete."
    }
}

agent = lmai.agents.SQLAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

## Advanced configuration

### View and debug prompts

**Print the default prompt:**

See what the original template looks like:

```python
import lumen.ai as lmai

agent = lmai.agents.ChatAgent()
print(agent.prompts["main"]["template"])
```

**Enable debug logging:**

See the rendered prompts sent to the LLM:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(
    data='penguins.csv',
    log_level='DEBUG'
)
ui.servable()
```

With DEBUG logging enabled, rendered prompts appear in the console output.

### Best practices

**Keep instructions clear.** Write concise, specific instructions. Vague instructions lead to unpredictable behavior.

**Use template_overrides first.** Start with `template_overrides` to modify specific blocks. Only replace entire templates if necessary.

**Test changes with your LLM.** Different models respond differently to prompt changes. Test with your production LLM model.

**Use {{ super() }} for extensions.** When you want to keep existing behavior and add to it, use `{{ super() }}` rather than replacing the entire block.

**Avoid hardcoding data.** Use memory variables instead of hardcoded values so prompts work with different data.

**Document custom templates.** If you create custom prompts, document what they expect from memory and how they differ from defaults.

**Enable DEBUG logging when testing.** Use `log_level='DEBUG'` to see the actual prompts being sent to the LLM.
