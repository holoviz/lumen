# :material-rocket: Quick Start

Install Lumen, set an API key, and start asking questions about your data in 30 seconds.

## Installation

```bash
pip install lumen
export OPENAI_API_KEY=sk-...
```

Using Anthropic, Gemini, Mistral, OpenRouter, a local model or a coding CLI you are already signed in to? The [Installation guide](installation.md) covers every provider.

!!! tip "No API key?"
    Skip the `export` and add `--provider kilo` to the command below. Kilo's free tier works without an account.

## Start Chatting with Data

=== "Command Line"

    ```bash
    lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv --show
    ```

=== "Python"

    ```python
    from lumen.ai import ExplorerUI
    
    ui = ExplorerUI(
        data='https://datasets.holoviz.org/penguins/v1/penguins.csv'
    )
    ui.servable()
    ```

    ```bash
    panel serve app.py --show
    ```

If a browser tab doesn't automatically open, visit [http://localhost:5006](http://localhost:5006) and start chatting with your data.

**Try these questions:**

- What datasets are available?

- Show me a summary of the data

- Which species has the largest average body mass? Show as a bar chart.

- Create a scatter plot of bill length vs flipper length, colored by island

- Filter for penguins over 4kg and show me the distribution by species

![Lumen Splash Screen](assets/navigating_the_ui/splash.png)

## How It Works

You don't need to write SQL or Python; just write your request:

> Which islands have the most penguins? Plot as a horizontal bar chart.

![Quick Start](assets/navigating_the_ui/results.png)

**1. Lumen creates a plan:**

- 🟢 Query the dataset to find the total number of penguins per island by grouping and summing the penguin counts for each island. Provide the resulting table with island names and penguin counts.
- 🟢 Create a horizontal bar chart to visualize the number of penguins per island using the table data provided by the previous step. The x-axis should represent the number of penguins and the y-axis the island names.
- 🟢 Validate whether the executed plan fully answered the user's original query.

**2. Generates SQL to query the data:**

```sql
SELECT "island", COUNT(*) AS "penguin_count" 
FROM penguins 
GROUP BY "island" 
ORDER BY "penguin_count" DESC
```

| island    | penguin_count |
|-----------|---------------|
| Biscoe    | 168           |
| Dream     | 124           |
| Torgersen | 52            |

**3. Creates a Vega-Lite visualization:**

```yaml
$schema: https://vega.github.io/schema/vega-lite/v5.json
data:
  name: penguin_count_by_island
height: container
layer:
- encoding:
    color:
      value: '#4682b4'
    x:
      axis:
        title: Penguin Count
      field: penguin_count
      type: quantitative
    y:
      axis:
        title: Island
      field: island
      sort: -x
      type: nominal
  mark: bar
title:
  anchor: start
  fontSize: 20
  subtitle: Biscoe island has the highest penguin count, followed by Dream and Torgersen
  subtitleColor: '#666666'
  subtitleFontSize: 16
  text: Penguin Counts by Island
width: container
```

**4. Renders the result:**

<iframe 
  src="../assets/penguin_counts_by_island.html" 
  width="100%" 
  height="600"
  style="border:none;"
></iframe>

All of this happens automatically when you just ask a question.

## Your Work is Kept During Your Session

Behind the scenes, Lumen organizes your work into **Explorations** — think of them as workspaces for each dataset or analysis thread. Explorations last for your browser session; they are not saved when you close or reload the page, so export anything you want to keep.

When you ask your first question that queries data, Lumen creates an exploration to capture everything:

- Your conversation history
- The SQL queries generated
- Charts and visualizations
- Results and data tables

**Follow-up questions stay together:** Ask "Can you make that chart show only the top 5?" and your new chart appears in the same exploration.

**New topics create new explorations:** Ask about a different dataset or unrelated question, and Lumen starts a fresh exploration.

This means you can:

- Return to any earlier analysis in the same session
- Export your work as a Jupyter notebook
- Keep multiple investigations organized
- Build on previous results without losing context

## Try with Your Own Data

=== "CSV Files"
    ```bash
    lumen serve data/sales.csv
    lumen serve https://example.com/data.csv
    ```

=== "Databases"
    ```bash
    lumen serve postgresql://user:pass@localhost/mydb
    lumen serve mysql://user:pass@localhost/mydb
    lumen serve sqlite:///data.db
    ```

## Next Steps

- **[Navigating the UI](getting_started/navigating_the_ui.md)** - Learn the chat interface
- **[Using Lumen AI](getting_started/using_lumen_ai.md)** - Master natural language queries
- **[Examples](examples/tutorials/index.md)** - Step-by-step tutorials
- **[Configure LLM Providers](configuration/llm_providers.md)** - Customize your AI model
