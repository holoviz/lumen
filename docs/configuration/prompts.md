# :material-message-text: Prompts

Prompts control what agents say and how they behave.

Most users don't need to customize prompts. Customize only if agents consistently make mistakes you can fix with instructions.

## How Prompts Work

Lumen uses Jinja2 template inheritance. All agents extend from `Actor/main.jinja2`:

```jinja2
{% extends 'Actor/main.jinja2' %}
```

This inheritance is why `{{ super() }}` works—it calls the parent block's content.

## Quick examples

### Change agent tone

``` py title="Add personality"
import lumen.ai as lmai

instructions = """
{{ super() }}

Be warm and enthusiastic.
"""

template_overrides = {
    "main": {
        "instructions": instructions
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

`{{ super() }}` keeps original instructions and adds yours after.

### Add SQL rules

``` py title="SQL guidelines"
instructions = """
{{ super() }}

Additional rules:
- Use explicit JOIN syntax
- Format dates as YYYY-MM-DD
- Use meaningful table aliases
"""

template_overrides = {
    "main": {
        "instructions": instructions
    }
}

agent = lmai.agents.SQLAgent(template_overrides=template_overrides)
```

### Add domain knowledge

``` py title="Domain context"
context = """
{{ super() }}

In our database:
- "Accounts" means customer accounts
- Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
"""

template_overrides = {
    "main": {
        "context": context
    }
}
```

## Override blocks

### Block Execution Order

Blocks render in this order (top to bottom = what the LLM sees):

| # | Block | Purpose | Auto-populated? | Common Use |
|---|-------|---------|-----------------|------------|
| 1 | `global` | Domain knowledge for all agents | No | Rarely used |
| 2 | `datetime` | Current timestamp | Yes | Keep default |
| 3 | `instructions` | Main task rules | No | **Most common** |
| 4 | `examples` | Example outputs | No | **Common** |
| 5 | `tools` | Tool-specific contexts | Yes | Rarely override |
| 6 | `context` | Agent-specific knowledge | No | **Common** |
| 7 | `errors` | Previous errors to fix | Yes | Rarely override |
| 8 | `footer` | Closing notes | No | Occasional |

**Auto-populated blocks:**

- `datetime`: Always includes current time
- `tools`: Populated from `memory["agent_tool_contexts"]` and `tool_context`
- `errors`: Only appears when previous execution failed (shows `last_output` and error messages)

### Most Useful Blocks

Three blocks cover most needs:

- `instructions` - Main task rules
- `examples` - Show desired output format
- `context` - Add domain knowledge

## Multiple agents

``` py title="Customize multiple agents"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[
        lmai.agents.ChatAgent(template_overrides=chat_overrides),
        lmai.agents.AnalystAgent(template_overrides=analyst_overrides),
    ]
)
ui.servable()
```

## Agents with multiple prompts

Most agents have one prompt. Some have more:

- **SQLAgent**: `main`, `select_discoveries`, `check_sufficiency`, `revise_output`
- **VegaLiteAgent**: `main`, `interaction_polish`, `annotate_plot`, `revise_output`

Override specific prompts:

``` py title="Multiple prompt overrides" hl_lines="2-5"
main_instructions = "{{ super() }} Generate optimized SQL."
revise_instructions = "{{ super() }} When fixing errors, explain what went wrong."

template_overrides = {
    "main": {
        "instructions": main_instructions
    },
    "revise_output": {
        "instructions": revise_instructions
    }
}
```

## Replace entire prompt

Only do this if block overrides don't work:

``` py title="Full prompt replacement"
full_template = """
You are a retail analytics assistant.

Data: {{ memory['data'] }}
Question: {{ messages[-1]['content'] }}

Focus on customer segments and purchase patterns. Be concise.
"""

prompts = {
    "main": {
        "template": full_template
    }
}

agent = lmai.agents.ChatAgent(prompts=prompts)
```

!!! warning "You lose all defaults"
    Full replacement discards all built-in instructions. Use block overrides with `{{ super() }}` instead.

## Load prompts from files

For complex prompts, load from external files using absolute paths:

``` py title="Load from file (absolute path)"
from pathlib import Path

template_overrides = {
    "main": {
        "instructions": str(Path(__file__).parent / 'agents.py')
    }
}

agent = lmai.agents.SQLAgent(template_overrides=template_overrides)
```

Create `agents.py` with your instructions:

``` py title="agents.py"
Your SQL agent handles customer data queries.

Rules:
- Always use INNER JOIN for relationships
- Sanitize date inputs to YYYY-MM-DD format
- Group by customer segments first

Examples:
- "Top customers" → Order by revenue DESC
- "Monthly trends" → Use DATE_TRUNC
```

!!! tip "Use absolute paths"
    Always use absolute paths (e.g., `/home/user/prompts/agents.py` or `Path(__file__).parent / 'agents.py'`) to avoid issues with working directory changes.

## Global context for all agents

Add domain knowledge visible to all agents:

``` py title="Global template overrides" hl_lines="1-8"
global_context = """
{{ super() }}

Domain knowledge:
- Inversions occur when temperature increases with altitude
- Standard lapse rate is 6.5°C per km
"""

# Apply to base Actor class
lmai.actor.Actor.template_overrides = {
    "main": {
        "global": global_context
    }
}

ui = lmai.ExplorerUI(data='weather.csv')
ui.servable()
```

## Debug prompts

See what the LLM receives:

``` py title="Enable debug logging"
ui = lmai.ExplorerUI(data='penguins.csv', log_level='DEBUG')
ui.servable()
```

Check console for full prompts sent to the LLM.

## Advanced: Override All Blocks

For complete control over prompt structure:

``` py title="All blocks"
global_block = "{{ super() }} Shared across all agents"
datetime_block = "{{ super() }}"  # Keep default
instructions_block = """
{{ super() }}

Additional rules here
"""
examples_block = """
{{ super() }}

More examples here
"""
tools_block = "{{ super() }} Tool-specific notes"
context_block = """
{{ super() }}

Domain-specific knowledge
"""
errors_block = "{{ super() }} Custom error handling"
footer_block = "Remember to be concise and accurate."

template_overrides = {
    "main": {
        "global": global_block,
        "datetime": datetime_block,
        "instructions": instructions_block,
        "examples": examples_block,
        "tools": tools_block,
        "context": context_block,
        "errors": errors_block,
        "footer": footer_block
    }
}
```

!!! warning "Don't override auto-populated blocks unnecessarily"
    The `datetime`, `tools`, and `errors` blocks are auto-populated. Only override if you need custom behavior.

## Troubleshooting

**Agent ignores instructions** - Be more specific. Use examples instead of rules.

**`{{ super() }}` causes errors** - Only use `{{ super() }}` when the parent block has content. Empty blocks in `Actor/main.jinja2`: `global`, `instructions`, `examples`, `context`, `footer`.

**Works with one LLM, not another** - Different LLMs need different prompt styles. Test with your production model.

## Best practices

- Start with `{{ super() }}` to keep defaults
- Be specific: "Keep responses under 3 sentences" not "Be concise"
- Use examples over rules
- Test with `log_level='DEBUG'`
- Only customize when defaults consistently fail
