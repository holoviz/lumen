# Chat with Your Data

Lumen is an open-source AI agent framework for interactive data analysis. Ask questions about your data—Lumen generates SQL, runs it, and shows you charts. No coding needed.

## What Makes Lumen Different

| Feature | Lumen | Other Tools |
|---------|-------|-----------|
| **Your LLM** | Use any LLM (yours or cloud) | Locked to one provider |
| **Your data** | Any source: local, DB, API, lake | Limited connectors |
| **Extend it** | Build agents, tools, visualizations | Limited customization |
| **Open source** | 100% open, no telemetry | Often proprietary |
| **Declarative** | YAML + Python, easily reproducible | Often code-only |

## Get Started in 30 Seconds

Using one of the methods below, start Lumen with a sample dataset, open [https://localhost:5006](https://localhost:5006), and start asking questions!

**Command line:**

```bash
pip install 'lumen[ai-openai]'  # or see Installation guide for other LLMs
lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv
```

---

**Python:**

Create `app.py`:

```python
from lumen.ai import ExplorerUI
ui = ExplorerUI(data='https://datasets.holoviz.org/penguins/v1/penguins.csv')
ui.servable()
```

Then run:

```bash
panel serve app.py
```

!!! tip
    You may need to specify an API key for your chosen LLM provider, e.g. `export OPENAI_API_KEY=your_key_here` for OpenAI. See [LLM Providers](configuration/llm_providers.md) for details.

## How It Works

You don't need to write SQL or Python; just ask a question, e.g.

→ **Which islands have the most penguins? Plot as a horizontal bar chart.**

Lumen creates a plan:

```markdown
- [ ] Query the penguin dataset to aggregate the total number of penguins by island. Provide the pipeline and with the aggregated data.
- [ ] Using the aggregated penguin count by island, create a horizontal bar chart showing islands on the y-axis and penguin counts on the x-axis.
```

From the plan, Lumen generates SQL to query the data:

```sql
SELECT "island", COUNT(*) AS "penguin_count" FROM penguins GROUP BY "island" ORDER BY "penguin_count" DESC
```

| island    | penguin_count |
|-----------|--------------|
| Biscoe    | 168          |
| Dream     | 124          |
| Torgersen | 52           |

Then Lumen creates a Vega-Lite spec to visualize the results:

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

<iframe 
  src="assets/penguin_counts_by_island.html" 
  width="640" 
  height="620"
  style="border:none;"
></iframe>

---

## Community & Support

- **Questions?** [Discourse](https://discourse.holoviz.org/c/lumen/)
- **Chat with us** [Discord](https://discord.com/invite/rb6gPXbdAr)
- **Found a bug?** [GitHub Issues](https://github.com/holoviz/lumen/issues)
- **Want to contribute?** [Contributing guide](contributing.md)
