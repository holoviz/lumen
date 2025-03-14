# Template Prompts

## Overview

Every `Actor` (`Coordinator`, `Agent`, `Tool`) operates based on a system prompt structured with modular components:

1. `instructions`
2. `context`
3. `tools`
4. `examples`

The base system prompt template follows this format:

```jinja2
{% block instructions %}
{% endblock %}

{% block context %}
{% endblock %}

{% block tools %}
{%- if 'tool_context' in memory -%}
{#- Context from Coordinator.tools -#}
{{ memory["tool_context"] }}
{%- endif -%}
{%- if tool_context is defined -%}
{{ tool_context }}
{% endif -%}
{% endblock %}

{% block examples %}
{% endblock %}
```

## Override Blocks

For example, the `ChatAgent`'s prompt template uses `instructions` and `context`:

```jinja2
{% extends 'Agent/main.jinja2' %}

{% block instructions %}
Act as a helpful assistant for high-level data exploration, focusing on available datasets and, only if data is
available, explaining the purpose of each column. Offer suggestions for getting started if needed, remaining factual and avoiding speculation. Do not write code or give code related suggestions.
{% endblock %}

{% block context %}
{% if 'data' in memory %}
Here's a summary of the dataset the user just asked about:
\```
{{ memory['data'] }}
\```
{% endif %}
{% if tables_schemas is defined %}
Available tables:
{% for table, schema in tables_schemas %}
- **{{ table }}** with schema: {{ schema }}
{% endfor %}
{% elif table is defined and schema is defined %}
{{ table }} with schema: {{ schema }}
{% endif %}
{% endblock %}
```

Here, the instructions are:

"""
Act as a helpful assistant for high-level data exploration, focusing on available datasets and, only if data is
available, explaining the purpose of each column. Offer suggestions for getting started if needed, remaining factual and avoiding speculation. Do not write code or give code related suggestions.
"""

If you'd like to override this you can specify `template_overrides`:

```python
template_overrides = {
    "main": {
        "instructions": "Act like the user's meteorologist, and explain jargon in the format of a weather report."
    },
}
agents = [lmai.agents.ChatAgent(template_overrides=template_overrides)]
ui = lmai.ExplorerUI(agents=agents)
```

This will result in the following prompt template:

```jinja2
{% extends 'Agent/main.jinja2' %}

{% block instructions %}
Act like the user's meteorologist, and explain jargon in the format of a weather report.
{% endblock %}

{% block context %}
{% if tables|length > 1 %}
Available tables:
{{ closest_tables }}
{% elif schema %}
{{ table }} with schema: {{ schema }}
{% endif %}
{% if 'data' in memory %}
Here's a summary of the dataset the user recently inquired about:
\```
{{ memory['data'] }}
\```
{% endif %}
{% endblock %}
```

:::{admonition} Tip
:class: success

Some `Agent`s may have multiple prompts, besides the `main` prompt. For example, `SQLAgent` has:

- `main`: The main prompt that generates the SQL query
- `select_table`: Prompt to select the most relevant table
- `require_joins`: Prompt to determine whether joins are required
- `find_joins`: Prompt to find the necessary joins

Not all prompts will be used in every interaction, e.g. `find_joins` may not be used if `require_joins` decides that joins are not necessary.

To see all the prompts available for an `Agent`, you can check the `prompts` parameter of the `Agent` class.
:::

If you simply want to prefix or suffix the original `instructions`, you can specify `{{ super() }}`:

```python
template_overrides = {
    "main": {
        "instructions": "{{ super() }}. Spice it up by speaking like a pirate."
    },
}
```

If you aren't sure what the original prompt is, you can print it out:

```python
print(Agent.get_prompt_template())  # defaults to key='main'
```

:::{admonition} Tip
:class: success

To debug prompts, you may specify `log_level="DEBUG"` to see the rendered prompts.
:::

You can also provide `examples`:

```python
template_overrides = {
    "main": {
        "instructions": "Speak like a pirate.",
        "examples": """
            Examples:
            '''
            Yarr, the wind be blowin' from the north at 10 knots.
            '''

            '''
            Arr, the temperature be 80 degrees Fahrenheit.
            '''
        """
    },
}
agents = [lmai.agents.ChatAgent(template_overrides=template_overrides)]
ui = lmai.ExplorerUI(agents=agents)
```

Which produces the following prompt template:

```jinja2
{% extends 'Agent/main.jinja2' %}

{% block instructions %}
Speak like a pirate.
{% endblock %}

{% block context %}
{% if tables|length > 1 %}
Available tables:
{{ closest_tables }}
{% elif schema %}
{{ table }} with schema: {{ schema }}
{% endif %}
{% if 'data' in memory %}
Here's a summary of the dataset the user recently inquired about:
\```
{{ memory['data'] }}
\```
{% endif %}
{% endblock %}

{% block examples %}
Example:
'''
Yarr, the wind be blowin' from the north at 10 knots.
'''
{% endblock %}
```

### Replace Template

Alternatively, if you'd like to replace the entire prompt template, you can specify the `template` key in `prompts` as a string or a valid path to a template:

```python
prompts = {
    "main": {
        "template": """
            Act like the user's meteorologist, and explain jargon in the format of a weather report.

            Available tables:
            {closest_tables}

            {table} with schema: {schema}
        """
    }
}
agents = [lmai.agents.ChatAgent(prompts=prompts)]
ui = lmai.ExplorerUI(agents=agents)
```

:::{admonition} Warning
:class: warning

If you override the prompt template, ensure that the template includes all the necessary parameters. If any parameters are missing, the LLM may lack context and provide irrelevant responses.
:::

For a listing of prompts, please see the [Lumen codebase](https://github.com/holoviz/lumen/tree/main/lumen/ai/prompts).
