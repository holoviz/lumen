# Using Lumen AI

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

If you want to see how the AI arrived at an answer, click on the chat steps' cards to expand them and view the LLM's reasoning (Chain of Thought).

## Refine results

If results aren't what you expected, you have several options:

**Retry with feedback** — Click the retry button and provide feedback. Lumen will send your previous messages, the last output, and your feedback to a larger, more capable model to regenerate the response.

**Continue the conversation** — Send a new message to refine or adjust the results. This sends a fresh request to the LLM.

**Manually edit** — Directly edit the SQL query or Lumen specification to get the desired output. This works if you're comfortable with SQL or Lumen syntax.

Use manual editing for small tweaks (like changing chart sort order), and send a new message for bigger changes to the underlying query.

## Explorations

Each new computed result creates a new "Exploration" tab. Click the Explorations panel (right sidebar) to navigate between them and compare different analyses.

## Export results

Export your session as a Jupyter notebook so you can reproduce, share, or build on your work.

**Export all explorations:**
Click **Export Notebook** at the top of the screen to download a notebook containing all your questions, queries, and visualizations.

**Export one exploration:**
Click the exploration-specific download button (visible in the top-right when viewing that exploration) to export only that analysis.

The notebook includes:
- Markdown cells with your inputs and outputs from agents like ChatAgent and AnalystAgent
- Code cells with Lumen specifications used to generate visualizations and tables

## Advanced options

### Command-line configuration

Pass additional options to the `serve` command:

```bash
lumen-ai serve --agents TableListAgent ChatAgent --help
```

Run `lumen-ai serve --help` to see all available options.

Agent names are case-insensitive and the suffix "agent" can be dropped (e.g., `ChatAgent` can be specified as `chatagent` or `chat`).

### Python API configuration

For fine-grained control, use the Python API to configure agents, tools, and other settings:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(
    data='data.csv',
    # ... other configuration options
)
ui.servable()
```

See [Agents](../configuration/agents.md), [LLM Providers](../configuration/llm_providers.md), and [Prompts](../configuration/prompts.md) configuration guides for all available options.

## Next steps

Learn more about configuration and advanced use cases:

- [Agents](../configuration/agents.md) — Use and customize agents
- [LLM Providers](../configuration/llm_providers.md) — Configure your LLM
- [Prompts](../configuration/prompts.md) — Customize agent prompts
