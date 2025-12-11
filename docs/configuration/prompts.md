# :material-message-text: Prompts

Prompts control what agents say and how they behave.

Most users don't need to customize prompts. Customize only if agents consistently make mistakes you can fix with instructions.

## Quick examples

### Change agent tone

``` py title="Add personality"
import lumen.ai as lmai

template_overrides = {
    "main": {
        "instructions": "{{ super() }} Be warm and enthusiastic."
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='penguins.csv', agents=[agent])
ui.servable()
```

`{{ super() }}` keeps original instructions and adds yours after.

### Add SQL rules

``` py title="SQL guidelines"
template_overrides = {
    "main": {
        "instructions": """{{ super() }}

Additional rules:
- Use explicit JOIN syntax
- Format dates as YYYY-MM-DD
- Use meaningful table aliases
"""
    }
}

agent = lmai.agents.SQLAgent(template_overrides=template_overrides)
```

### Add domain knowledge

``` py title="Domain context"
template_overrides = {
    "main": {
        "context": """{{ super() }}

In our database:
- "Accounts" means customer accounts
- Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
"""
    }
}
```

## Override blocks

Three blocks cover most needs:

| Block | Use for |
|-------|---------|
| `instructions` | Main task rules |
| `examples` | Show desired output format |
| `context` | Add domain knowledge |

Other blocks exist (`errors`, `datetime`, `footer`, `tools`, `global`) but are rarely needed.

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
template_overrides = {
    "main": {
        "instructions": "Generate optimized SQL."
    },
    "revise_output": {
        "instructions": "When fixing errors, explain what went wrong."
    }
}
```

## Replace entire prompt

Only do this if block overrides don't work:

``` py title="Full prompt replacement"
prompts = {
    "main": {
        "template": """
You are a retail analytics assistant.

Data: {{ memory['data'] }}
Question: {{ messages[-1]['content'] }}

Focus on customer segments and purchase patterns. Be concise.
"""
    }
}

agent = lmai.agents.ChatAgent(prompts=prompts)
```

!!! warning "You lose all defaults"
    Full replacement discards all built-in instructions. Use block overrides with `{{ super() }}` instead.

## Global context for all agents

Add domain knowledge visible to all agents:

``` py title="Global template overrides" hl_lines="1-7"
global_context = """
{{ super() }}

Domain knowledge:
- Inversions occur when temperature increases with altitude
- Standard lapse rate is 6.5°C per km
"""

# Apply to base Actor class
lmai.actor.Actor.template_overrides = {"main": {"global": global_context}}

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

## Troubleshooting

**Agent ignores instructions** - Be more specific. Use examples instead of rules.

**`{{ super() }}` causes errors** - Only use in standard blocks (`instructions`, `examples`, `context`).

**Works with one LLM, not another** - Different LLMs need different prompt styles. Test with your production model.

## Best practices

- Start with `{{ super() }}` to keep defaults
- Be specific: "Keep responses under 3 sentences" not "Be concise"
- Use examples over rules
- Test with `log_level='DEBUG'`
- Only customize when defaults consistently fail
