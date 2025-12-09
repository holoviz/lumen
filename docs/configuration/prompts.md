# Prompts

**Prompts control what agents say and how they behave.**

Most users don't need to customize prompts. Customize only if agents consistently make mistakes you can fix with instructions, or you need specific terminology.

## Quick examples

### Change agent tone

```python
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

```python
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

### Add examples

```python
template_overrides = {
    "main": {
        "examples": """
Good response: "Widget X leads with $1.2M (45% of total). Top 2 products account for 75% of revenue."

Always include:
- Specific numbers with units
- Percentages for context
- One actionable insight
"""
    }
}

agent = lmai.agents.AnalystAgent(template_overrides=template_overrides)
```

### Add domain knowledge

```python
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

## What you can override

Three blocks cover most needs:

| Block | Use for |
|-------|---------|
| `instructions` | Main task rules |
| `examples` | Show desired output format |
| `context` | Add domain knowledge |

Other blocks exist (`errors`, `datetime`, `footer`) but are rarely needed.

## Multiple agents

```python
ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[
        lmai.agents.ChatAgent(template_overrides=chat_overrides),
        lmai.agents.AnalystAgent(template_overrides=analyst_overrides),
        lmai.agents.SQLAgent(template_overrides=sql_overrides),
    ]
)
ui.servable()
```

## Agents with multiple prompts

Most agents have one prompt. Some have more:

- **SQLAgent**: `main`, `revise_output`, `select_discoveries`, `check_sufficiency`
- **VegaLiteAgent**: `main`, `revise_output`, `improvement_step`

Override specific ones:

```python
template_overrides = {
    "main": {
        "instructions": "Generate optimized SQL."
    },
    "revise_output": {
        "instructions": "When fixing errors, explain what went wrong."
    }
}
```

## Replace entire prompt (rare)

Only do this if block overrides don't work:

```python
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

**Warning:** You lose all default instructions. Use block overrides instead when possible.

## Debug prompts

See what the LLM receives:

```python
ui = lmai.ExplorerUI(data='penguins.csv', log_level='DEBUG')
ui.servable()
```

Check console for full prompts sent to the LLM.

## Troubleshooting

**Agent ignores instructions**: Be more specific. Use examples instead of rules.

**{{ super() }} causes errors**: Only use in standard blocks (`instructions`, `examples`, `context`).

**Works with one LLM, not another**: Different LLMs need different prompt styles. Test with your production model.

**Not sure if it's working**: Enable `log_level='DEBUG'` to see actual prompts.

## Best practices

- Start with `{{ super() }}` to keep defaults
- Be specific ("Keep responses under 3 sentences" not "Be concise")
- Use examples over rules
- Test with `log_level='DEBUG'`
- Only customize when defaults consistently fail

## When NOT to customize

Don't customize if:
- You haven't tried defaults yet
- Agent makes occasional mistakes (not consistent)
- You want to change what the agent does (create a custom agent instead)
- You need a better model (upgrade your LLM instead)
