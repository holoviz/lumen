# :material-chat: Chat with Your Data

**Open-source, fully extensible Python framework for AI-powered data exploration. Easy to use, reproducible results, and tailored to your domain expertise.**

https://github.com/user-attachments/assets/ef6565f3-a21a-4a3e-b862-0b2b539b3261

Ask questions in plain English and get instant answers, visualizations, and reports. No coding required. Natural language queries automatically generate database queries, create visualizations, and build dashboards. **[Get started in 30 seconds →](quick_start.md)**

Bring your own LLM: OpenAI, Anthropic, local models. Connect to your data sources: Snowflake, BigQuery, DuckDB, and any database supported by SQLAlchemy. **[See all installation options →](installation.md)**

=== "CSV File + OpenAI"

    ```bash
    export OPENAI_API_KEY=your_key_here
    lumen-ai serve your_data.csv --provider openai --show
    ```
    
    **Then ask:**
    
    > "Show me the top 10 rows"
    
    > "What's the average revenue by region? Plot as a bar chart."
    
    > "Filter for sales over $1000 and create a histogram"

=== "Snowflake + Mistral"

    ```python
    from lumen.sources.snowflake import SnowflakeSource
    import lumen.ai as lmai

    source = SnowflakeSource(
        account='your-account',
        database='your-database',
        authenticator='externalbrowser'  # SSO login
    )

    llm = lmai.llm.MistralAI()  # Enterprise-grade EU LLM
    ui = lmai.ExplorerUI(data=source, llm=llm)
    ui.servable()
    ```
    
    **Then ask:**
    
    > "Which customers have the highest lifetime value?"
    
    > "Show monthly revenue trends for the last year"
    
    > "Join orders and products, then show top categories by profit"

=== "BigQuery + Google"

    ```python
    from lumen.sources.bigquery import BigQuerySource
    import lumen.ai as lmai

    source = BigQuerySource(project_id='your-project-id')

    llm = lmai.llm.Google()  # Native Google AI integration
    ui = lmai.ExplorerUI(data=source, llm=llm)
    ui.servable()
    ```
    
    **Then ask:**
    
    > "What are the peak traffic hours? Show as a line chart."
    
    > "Calculate conversion rate by source and display in a table"
    
    > "Find anomalies in daily user counts"

=== "DuckDB + Anthropic"

    ```python
    from lumen.sources.duckdb import DuckDBSource
    import lumen.ai as lmai

    source = DuckDBSource(
        tables={
            "penguins": "https://datasets.holoviz.org/penguins/v1/penguins.csv"
        }
    )

    llm = lmai.llm.Anthropic()  # Claude for complex reasoning
    ui = lmai.ExplorerUI(data=source, llm=llm)
    ui.servable()
    ```
    
    **Then ask:**
    
    > "Which islands have the most penguins? Plot as a horizontal bar chart."
    
    > "Create a scatter plot of bill length vs body mass, colored by species"
    
    > "Calculate the ratio of flipper length to body mass, then plot it"

=== "PostgreSQL + Ollama (Local)"

    ```bash
    # Run 100% local - no cloud, full privacy
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=ollama
    
    lumen-ai serve postgresql://user:pass@localhost/mydb \
      --provider ollama \
      --model qwen3:8b \
      --show
    ```
    
    **Then ask:**
    
    > "Show me customer signup trends by month"
    
    > "Which products have declining sales? Show the trend."
    
    > "Find users who haven't logged in for 90 days"

---

## Why Lumen?

**Stop waiting for data teams. Ask questions in plain English.**

You have questions about your data. Your data team is backlogged three sprints. You could learn SQL, but by the time you do, the question will have changed.

Lumen eliminates this bottleneck. **[See launch options →](getting_started/launching_lumen.md)**

| Who | Benefit |
|----|---------|
| **Data analysts** | Skip boilerplate code and focus on insights |
| **Developers** | Build data apps in hours instead of weeks |
| **Data teams** | Enable self-service without losing control |
| **Researchers** | Share reproducible analysis without teaching syntax |

**No vendor lock-in.** Use OpenAI, Anthropic, or run models locally with LlamaCPP so that your data never leaves your machine. **[Configure your LLM provider →](configuration/llm_providers.md)**

```python
# Run 100% local - no API keys, no cloud, full privacy
llm = lmai.llm.LlamaCpp(
    model_kwargs={
        'default': {
            'repo_id': 'unsloth/Qwen3-8B-GGUF',
            'filename': 'Qwen3-8B-Q5_K_M.gguf'
        }
    }
)
ui = lmai.ExplorerUI(data='data.csv', llm=llm)

# Or switch to cloud providers anytime - just one line
# llm=lmai.llm.OpenAI() or Anthropic(), Google(), MistralAI()
```

**Connect to anything.** PostgreSQL, Snowflake, DuckDB, CSV files, APIs—if it has data, Lumen can support it. **[See all data sources →](configuration/sources.md)**

```python
# Mix sources - local files + cloud databases
ui = lmai.ExplorerUI(data=[
    'local_data.csv',                                    # CSV file
    'postgresql://localhost/mydb',                       # PostgreSQL
    SnowflakeSource(account='...', database='...'),      # Snowflake
    BigQuerySource(project_id='my-project')              # BigQuery
])
```

**Infinitely extensible.** Pure Python + YAML configs. Build custom agents, tools, and visualizations when you need them. **[View examples →](examples/tutorials/index.md)**

=== "Custom Analysis"

    ```python
    # Add domain-specific analysis in minutes
    class RevenueAnalysis(lmai.analysis.Analysis):
        """Calculate key revenue metrics."""

        columns = ['revenue', 'churned']

        def __call__(self, pipeline, ctx):
            df = pipeline.data
            return pd.DataFrame({
                'MRR': df['revenue'].sum() / 12,
                'Churn Rate': (df['churned'].sum() / len(df)) * 100
            }, index=[0])

    ui = lmai.ExplorerUI(data=source, analyses=[RevenueAnalysis])
    ```
    
    Analyses run deterministic calculations with code instead of LLM generation—perfect for financial metrics, scientific formulas, or any calculation that must produce identical results every time.
    
    **[Learn about analyses →](configuration/analyses.md)**

=== "Customize Agent Prompts"

    ```python
    # Teach agents your domain terminology
    context = """
    {{ super() }}
    
    In our business:
    - "Accounts" means customer accounts
    - Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
    - "Active" means logged in within 30 days
    """
    
    agent = lmai.agents.ChatAgent(
        template_overrides={"main": {"context": context}}
    )
    ui = lmai.ExplorerUI(data=source, agents=[agent])
    ```
    
    Agents are specialized workers that handle different question types (SQLAgent writes queries, VegaLiteAgent creates charts). Customize their prompts to teach them your domain terminology and business rules.
    
    **[Learn prompt customization →](configuration/prompts.md)**

=== "Add Custom Tools"

    ```python
    # Extend agents with custom functions
    def calculate_customer_ltv(customer_id: str, table) -> dict:
        """Calculate customer lifetime value."""
        customer_data = table[table['customer_id'] == customer_id]
        total_revenue = customer_data['revenue'].sum()
        return {
            'ltv': total_revenue,
            'avg_order': total_revenue / len(customer_data)
        }
    
    tool = lmai.tools.FunctionTool(
        function=calculate_customer_ltv,
        requires=['table'],
        provides=['ltv', 'avg_order']
    )
    ui = lmai.ExplorerUI(data=source, tools=[tool])
    ```
    
    Tools let agents access external data and perform specialized tasks—wrap APIs, integrate libraries, or add custom business logic that agents can call when needed.
    
    **[Learn about tools →](configuration/tools.md)**

---

## What Makes It Different?

**AI + YAML + Python. Use what fits your workflow.**

| Traditional BI | Code-First | Lumen |
|----------------|------------|--------|
| Inflexible, expensive | Too much boilerplate | Chat for instant results |
| Vendor lock-in | Manual UI building | YAML specs for reproducibility |
| No customization | Full control | Python extensions when needed |

Chat with your data to generate visualizations instantly, write YAML specs for reproducible dashboards, or drop into Python for custom logic:

=== "AI Conversation"

    **You:** "Create a dashboard showing penguin measurements by species with filters"
    
    **Lumen AI generates:**

    - SQL queries to aggregate data
    - Interactive filters for species, island, sex
    - Multiple visualizations (scatter plot, histogram, table)
    - Responsive layout
    
    **[Try it now →](quick_start.md)**

=== "YAML Specification"

    ```yaml
    sources:
      penguins:
        type: file
        tables:
          data: https://datasets.holoviz.org/penguins/v1/penguins.csv
    
    pipelines:
      filtered:
        source: penguins
        table: data
        filters:
          - type: widget
            field: species
          - type: widget
            field: island
    
    layouts:
      - title: Palmer Penguins Dashboard
        pipeline: filtered
        views:
          - type: hvplot
            kind: scatter
            x: bill_length_mm
            y: bill_depth_mm
            color: species
          - type: hvplot
            kind: hist
            y: body_mass_g
          - type: table
    ```
    
    **[Learn YAML specs →](examples/tutorials/penguins_dashboard_spec.md)**

=== "Result"

    ![Palmer Penguins Dashboard](assets/build_app_07.png)
    
    **Interactive dashboard with:**
    
    - 📊 Scatter plot showing bill measurements by species
    - 📈 Histogram of body mass distribution
    - 📋 Filterable data table
    - 🎛️ Sidebar widgets for filtering
    - 🎨 Dark theme
    - 📱 Responsive design
    
    Built in 15 minutes with YAML or 2 minutes with AI chat.

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
