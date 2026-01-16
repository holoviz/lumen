# :material-brain: Using Lumen AI

Once you're set up and exploring, here's how to get the most out of Lumen AI.

## Message examples

Here are some ideas to spark your exploration:

**Basic queries:**

- "What datasets are available?"
- "Give me a summary of the dataset."
- "What are the columns in the dataset?"
- "What is the distribution of the 'species' column?"

**Visualizations:**

- "Show me a scatter plot of 'flipper_length_mm' vs 'body_mass_g'."
- "Create a histogram of 'bill_length_mm'."
- "Show me a bar chart of average values by species."

**Complex queries:**

- "Group by 'species' and show me the average 'flipper_length_mm'. Then plot a bar chart of the result."
- "Filter the dataset for species 'Chinstrap' and calculate the median 'body_mass_g'. Then display and discuss the result."
- "Create a histogram of 'bill_length_mm' and a box plot of 'flipper_length_mm' side by side."
- "Add a new column calculating the ratio of 'flipper_length_mm' to 'body_mass_g'. Then plot a scatter plot of this ratio against 'bill_length_mm'."
- "Select records where 'body_mass_g' is greater than 3500 and 'flipper_length_mm' is less than 200. Then provide summary statistics for these filtered records."

**Get inspired:**

- "What could be interesting to explore?"

## Combine multiple requests

You can ask the AI to perform several steps in one message. This helps you build complex analyses without multiple back-and-forths. For example: "Filter the data, create a visualization, then summarize the findings."

## Understand the AI's reasoning

If you want to see how the AI arrived at an answer, enable **Chain of Thought** in [**Settings**](navigating_the_ui.md#settings-and-control-options) (left sidebar). This will show the LLM's reasoning steps in expandable cards within the chat.

You can also configure your LLM provider and model from **Settings**.

## Refine results

If results aren't what you expected, you have several options:

**Rerun the query** — Click the **Rerun** button to re-execute the last query. This is useful if there was a temporary error or if you want to see if the AI produces different results.

**Continue the conversation** — Send a new message to refine or adjust the results. For example: "Can you make that chart show only the top 5 items?" or "Add a trend line to the visualization."

**Add annotations** — For visualizations, click the [annotation button](navigating_the_ui.md#revising-and-refining) (💬 icon) to add highlights, callouts, or labels. For example: "Highlight the peak values" or "Mark outliers in red."

**Manually edit** — Directly edit the SQL query or visualization specification in the editor panel. This works if you're comfortable with SQL or need precise control over the output.

Use manual editing for small tweaks (like changing chart colors or sort order), and send a new message for bigger changes to the underlying query or analysis approach.

## Explorations

An Exploration is a persistent, contextual workspace for working with a specific dataset. It is created when a SQL query is first generated and captures the full interaction state, including the conversation, analyses, visualizations, and other data artifacts. An Exploration evolves over time, supports multiple questions and operations on the same data, and can be revisited or exported as a coherent unit.M

Explorations start from the global context (available sources and metadata). If a question is a follow-up, the new exploration is nested under the parent; if it is not, Lumen creates a new top-level exploration.

Use the [navigation menu](navigating_the_ui.md#understanding-explorations) to move between explorations or nest follow-ups under the exploration they build on.

## Report mode

**Report mode shows all your analyses on one page.** Switch to Report mode from the left sidebar to see your explorations organized into a structured report with collapsible sections.

**What you can do:**

- **Review everything** — Scroll through all analyses in one place
- **Edit prompts** — Click any task to modify its instructions and rerun
- **Customize agents** — Edit underlying agent prompts to change behavior
- **Export all** — Download everything as a single Jupyter notebook

**Under the hood:** Report mode uses Lumen's `Report` framework. Each exploration becomes a `Section` containing `ActorTask` objects. The same classes power both the UI and code-based reports.

!!! tip "Planned Feature: Report Templates"
    We're working on the ability to export your report structure as a reusable YAML configuration file. This will let you:

    - **Build a skeleton through exploration**: Ask questions naturally, and the system captures the underlying report structure (which agents, what order, what prompts).
    - **Create templated reports**: Export a "Q3 Customer Analysis" report, then next quarter reload it, change "Q3" to "Q4", and run—no re-prompting needed.
    - **Share report structures**: Export a config, share with colleagues, and they get the exact same report framework against their own data.
    - **Iterate without re-discovery**: Tweak the config (reorder sections, adjust prompts, swap one analysis for another) without starting from scratch.

    The LLM does the hard work once during exploration, then gets out of the way for repeated execution.

**Build reports programmatically:**

The Report framework can also be used directly in Python to create reusable analytical workflows:

```python
from lumen.ai.report import Action, Report, Section

class MyAnalysis(Action):
    async def _execute(self, context, **kwargs):
        # Your analysis here
        return [output], {'result': value}

report = Report(
    Section(MyAnalysis(title="Analysis"), title="Section"),
    title="My Report"
)
await report.execute()
```

## Export results

Export your session as a Jupyter notebook so you can reproduce, share, or build on your work.

**Export current exploration:**
Use **Export Notebook** in the navigation menu to download a notebook containing the current exploration's questions, queries, and visualizations.

**Export all explorations:**
Switch to **Report** mode (via the left sidebar), then use **Export Notebook** to download everything as a single notebook.

The notebook includes:

- Markdown cells with your questions and AI responses
- Code cells with SQL queries and Lumen specifications
- Visualizations as executable code

## Advanced options

### Command-line configuration

Pass additional options when launching Lumen AI:

``` bash title="Specify agents"
lumen-ai serve data.csv --agents SQLAgent VegaLiteAgent ChatAgent
```

``` bash title="Configure temperature"
lumen-ai serve data.csv --temperature 0.8
```

``` bash title="Use specific provider"
lumen-ai serve data.csv --provider anthropic --api-key sk-...
```

Run `lumen-ai serve --help` to see all available options.

!!! note "Agent names are flexible"
    Agent names are case-insensitive and the "Agent" suffix is optional: `SQLAgent` = `sqlagent` = `sql`

### Python API configuration

For fine-grained control, use the Python API:

```py title="Advanced Python configuration" hl_lines="4 6 7"
import lumen.ai as lmai

ui = lmai.ExplorerUI(
    data='data.csv',
    llm=lmai.llm.Anthropic(),  # (1)!
    default_agents=[lmai.agents.SQLAgent, lmai.agents.ChatAgent],
    log_level='INFO',
)
ui.servable()
```

1. Use Anthropic instead of default OpenAI

See the configuration guides for all available options:

- [Prompts](../configuration/prompts.md) — Customize agent behavior
- [Sources](../configuration/sources.md) — Connect to databases and files
- [LLM Providers](../configuration/llm_providers.md) — Configure your LLM
- [Agents](../configuration/agents.md) — Use and customize agents
- [Tools](../configuration/tools.md) — Extend capabilities with custom tools
- [Analyses](../configuration/analyses.md) — Add domain-specific analyses

## Next steps

Now that you know the basics, dive deeper into specific topics:

- [**Agents**](../configuration/agents.md) — Learn about the different agent types and how to customize them
- [**Prompts**](../configuration/prompts.md) — Fine-tune how agents respond
- [**Context**](../configuration/context.md) — Understand how agents share data
