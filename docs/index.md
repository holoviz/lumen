# :material-chat: Chat with Your Data

**Ask questions. Get answers, visualizations, and reports. No coding required.**

Lumen is an open-source Python framework for building AI-powered data tools.

Natural language queries automatically generate database queries, create visualizations, and build dashboards. Bring your own LLM—OpenAI, Anthropic, local models, or build your own. Infinitely extensible with pure Python.

=== "CSV File"

    ```bash
    lumen serve your_data.csv
    ```

=== "Snowflake"

    ```python
    from lumen.sources.snowflake import SnowflakeSource
    import lumen.ai as lmai

    source = SnowflakeSource(
        account='your-account',
        database='your-database',
        authenticator='externalbrowser'  # SSO login
    )

    ui = lmai.ExplorerUI(data=source)
    ui.servable()
    ```

---

## Why Lumen?

**Stop waiting for data teams. Ask questions in plain English.**

You have questions about your data. Your data team is backlogged three sprints. You could learn SQL, but by the time you do, the question will have changed.

Lumen eliminates this bottleneck:

| Who | Benefit |
|----|---------|
| **Data analysts** | Skip boilerplate code and focus on insights |
| **Developers** | Build data apps in hours instead of weeks |
| **Data teams** | Enable self-service without losing control |
| **Researchers** | Share reproducible analysis without teaching syntax |

**No vendor lock-in.** Use OpenAI, Anthropic, local Llama, or your own model. Self-host anywhere.

**Connect to anything.** PostgreSQL, Snowflake, DuckDB, CSV files, APIs—if it has data, Lumen can query it.

**Infinitely extensible.** Pure Python + YAML configs. Build custom agents, tools, and visualizations when you need them.

---

## What Makes It Different?

**Declarative + AI-powered + Bring your own LLM**

| Traditional BI | Code-First | Lumen |
|----------------|------------|--------|
| Inflexible, expensive | Too much boilerplate | YAML specs for speed |
| Vendor lock-in | Manual UI building | AI generates queries & viz |
| No customization | Full control | Python extensions when needed |

Write YAML for common patterns. Drop into Python for custom logic. AI handles the tedious parts.

---

## What's Next?

**Start with these guides:**

<div class="grid cards" markdown>

-   :material-rocket: **[Quick Start](quick_start.md)**  
    Get up and running in 30 seconds

-   :material-download: **[Installation](installation.md)**  
    Install with your preferred LLM provider

-   :material-play-circle: **[Launching Lumen](getting_started/launching_lumen.md)**  
    Command line and Python usage

-   :material-mouse: **[Navigating the UI](getting_started/navigating_the_ui.md)**  
    Chat interface and dashboard builder

-   :material-brain: **[Using Lumen AI](getting_started/using_lumen_ai.md)**  
    Master natural language queries

-   :material-school: **[Examples](examples/tutorials/index.md)**  
    Step-by-step tutorials and gallery

</div>

---

## Community & Support

Questions? Join our community:

- **Forum:** [Discourse](https://discourse.holoviz.org/c/lumen/)
- **Chat:** [Discord](https://discord.com/invite/rb6gPXbdAr)
- **Bugs:** [GitHub Issues](https://github.com/holoviz/lumen/issues)
- **Contributing:** [Guide](reference/contributing.md)

*[LLM]: Large Language Model
*[API]: Application Programming Interface
*[SQL]: Structured Query Language
*[YAML]: YAML Ain't Markup Language
*[CSV]: Comma-Separated Values
