# :material-rocket: Quick Start

Get up and running with Lumen in under 30 seconds.

## Installation

Install Lumen with your preferred LLM provider:

=== "OpenAI"

    ```bash
    pip install 'lumen[ai-openai]'
    export OPENAI_API_KEY=your_key_here
    ```

=== "Anthropic"

    ```bash
    pip install 'lumen[ai-anthropic]'
    export ANTHROPIC_API_KEY=your_key_here
    ```

=== "Local Models"

    ```bash
    pip install 'lumen[ai-local]'
    # Example: Using a local Ollama model
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=ollama  # Dummy key for local models
    ```

## Start Chatting with Data

=== "Command Line"

    ```bash
    lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv --show
    ```
    
    If a browser tab doesn't automatically open, visit [https://localhost:5006](https://localhost:5006) and start chatting with your data.
    
    **Try these questions:**
    
    > "What datasets are available?"
    
    > "Show me a summary of the data"
    
    > "Which species has the largest average body mass? Show as a bar chart."
    
    > "Create a scatter plot of bill length vs flipper length, colored by island"
    
    > "Filter for penguins over 4kg and show me the distribution by species"

=== "Python"

    ```python
    from lumen.ai import ExplorerUI
    
    ui = ExplorerUI(
        data='https://datasets.holoviz.org/penguins/v1/penguins.csv'
    )
    ui.servable()
    ```

    ```bash
    panel serve app.py
    ```
    
    **Then ask:**
    
    > "What's the correlation between bill length and body mass?"
    
    > "Group by island and show average measurements as a table"
    
    > "Create a histogram of flipper lengths with 20 bins"
    
    > "Are there any outliers in the body mass data?"

## How It Works

You don't need to write SQL or Python; just ask a question:

> **Which islands have the most penguins? Plot as a horizontal bar chart.**

**1. Lumen creates a plan:**

```markdown
- [ ] Query the penguin dataset to aggregate the total number of penguins by island.
- [ ] Create a horizontal bar chart showing islands on the y-axis and penguin counts on the x-axis.
```

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

All of this happens automatically when you just ask a question. Lumen groups the conversation and outputs from that work into an Exploration so you can keep building on it later.

## Explorations

An Exploration is a persistent, contextual workspace for working with a specific dataset. It is created when a SQL query is first generated and captures the full interaction state, including the conversation, analyses, visualizations, and other data artifacts. An Exploration evolves over time, supports multiple questions and operations on the same data, and can be revisited or exported as a coherent unit.

Explorations start from the global context (available sources and metadata). When you ask a question that triggers a SQL query, Lumen creates a new exploration; if the question is a follow-up, the new exploration is nested under the parent, and if it is not, Lumen creates a new top-level exploration.

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

=== "APIs & URLs"
    ```bash
    lumen serve https://api.example.com/data.json
    lumen serve https://datasets.holoviz.org/penguins/v1/penguins.csv
    ```

## Next Steps

- **[Navigating the UI](getting_started/navigating_the_ui.md)** - Learn the chat interface
- **[Using Lumen AI](getting_started/using_lumen_ai.md)** - Master natural language queries
- **[Examples](examples/tutorials/index.md)** - Step-by-step tutorials
- **[Configure LLM Providers](configuration/llm_providers.md)** - Customize your AI model
