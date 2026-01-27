# :material-file-document-multiple: Building AI-Powered Reports

Create multi-section analytical reports that combine deterministic analysis with AI-powered insights. Lumen Reports provide a structured framework for building reproducible, shareable analytics workflows.

**API Reference:** [Report API Docs](../../reference/report_api.md) — Complete class and method reference.

## Final result

<img width="1505" alt="Lumen Report Interface" src="https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/report-example.png" />

**Time**: 20-30 minutes

## What you'll build

An interactive subscription analytics report with three sections:

1. **Revenue Metrics** - Calculate MRR, analyze plan distribution
2. **Customer Health** - Track churn, cohort retention, and lifetime value
3. **AI Insights** - Generate natural language SQL queries and recommendations

The tutorial covers:

- Creating custom `Action` tasks for deterministic analysis
- Using `ActorTask` to wrap AI agents
- Passing context between tasks
- Organizing tasks into sections
- Exporting reports to Jupyter notebooks

## Prerequisites

Install the required packages:

```bash
pip install lumen-ai duckdb openai
```

Set your OpenAI API key (or configure another LLM provider):

```bash
export OPENAI_API_KEY='your-key-here'
```

## 1. Generate sample data

First, create a sample subscription database. This script generates realistic data with customers, subscriptions, payments, and churn events:

```python title="generate_database.py"
import duckdb
from datetime import datetime, timedelta
import random

# Connect to DuckDB
con = duckdb.connect('subscription_analytics.db')

# Create tables
con.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        email VARCHAR,
        name VARCHAR,
        signup_date DATE,
        status VARCHAR  -- 'Active' or 'Churned'
    )
""")

con.execute("""
    CREATE TABLE subscriptions (
        subscription_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        plan_type VARCHAR,  -- 'Basic', 'Pro', 'Enterprise'
        mrr DECIMAL(10,2),
        start_date DATE,
        end_date DATE,
        status VARCHAR,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
""")

con.execute("""
    CREATE TABLE churn_log (
        customer_id INTEGER,
        churn_date DATE,
        churn_reason VARCHAR,
        months_subscribed INTEGER,
        total_revenue DECIMAL(10,2)
    )
""")

# Generate sample data
plan_prices = {'Basic': 29.99, 'Pro': 99.99, 'Enterprise': 299.99}
customers_data = []

for i in range(1, 101):
    signup_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 700))
    status = 'Churned' if random.random() < 0.2 else 'Active'
    customers_data.append({
        'customer_id': i,
        'email': f'user{i}@example.com',
        'name': f'Customer {i}',
        'signup_date': signup_date,
        'status': status
    })

con.executemany(
    "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
    [(c['customer_id'], c['email'], c['name'], c['signup_date'], c['status']) 
     for c in customers_data]
)

print("✓ Database created: subscription_analytics.db")
con.close()
```

Run this script to create your database:

```bash
python generate_database.py
```

## 2. Create your first Action

Actions are the building blocks of Lumen Reports. They perform specific analyses and return both visual outputs and context data for downstream tasks.

**Key concepts:**

- **Actions** subclass `lumen.ai.report.Action` (see [Report API](../../reference/api.md))
- **`_execute()` method** contains the analysis logic
- **Return tuple** `(outputs, context)` where outputs are visualizations and context is data for other tasks
- **Context passing** allows tasks to share results

```python title="report.py" linenums="1"
from lumen.ai.report import Action, Report, Section
from panel.pane import Markdown
import duckdb

# Connect to database
con = duckdb.connect('subscription_analytics.db')

class MRRCalculation(Action):
    """Calculate Monthly Recurring Revenue metrics"""
    
    async def _execute(self, context, **kwargs):
        # Query the database
        result = con.execute("""
            SELECT 
                SUM(mrr) as total_mrr,
                COUNT(*) as active_subs,
                AVG(mrr) as avg_mrr
            FROM subscriptions
            WHERE status = 'Active'
        """).fetchone()
        
        total_mrr, active_subs, avg_mrr = result
        
        # Create markdown output
        output = Markdown(f"""
**Monthly Recurring Revenue**

- Total MRR: ${total_mrr:,.2f}
- Active Subscriptions: {active_subs}
- Average MRR: ${avg_mrr:.2f}
        """)
        
        # Return outputs and context
        return [output], {
            'total_mrr': total_mrr,
            'active_subs': active_subs,
            'avg_mrr': avg_mrr
        }  # (1)!
```

1. The context dictionary makes these values available to downstream tasks

### Test the Action

Before building the full report, test your Action independently:

```python
import asyncio

action = MRRCalculation(title="MRR Analysis")
outputs, context = asyncio.run(action.execute())

print(f"Total MRR: ${context['total_mrr']:,.2f}")
```

## 3. Add more analyses

Create additional Actions to build a comprehensive analysis. Each Action focuses on a specific metric:

```python title="report.py" linenums="40"
class ChurnAnalysis(Action):
    """Analyze customer churn patterns"""
    
    async def _execute(self, context, **kwargs):
        # Calculate churn rate
        churn_data = con.execute("""
            SELECT 
                COUNT(CASE WHEN status = 'Churned' THEN 1 END) as churned,
                COUNT(*) as total,
                ROUND(100.0 * COUNT(CASE WHEN status = 'Churned' THEN 1 END) / COUNT(*), 1) as rate
            FROM customers
        """).fetchone()
        
        churned, total, rate = churn_data
        
        # Get top churn reasons
        reasons = con.execute("""
            SELECT churn_reason, COUNT(*) as count
            FROM churn_log
            GROUP BY churn_reason
            ORDER BY count DESC
            LIMIT 3
        """).fetchall()
        
        output = Markdown(f"""
**Churn Analysis**

- Churn Rate: {rate}%
- Churned: {churned} / {total} customers

**Top Reasons:**
{chr(10).join(f'- {reason}: {count}' for reason, count in reasons)}
        """)
        
        return [output], {'churn_rate': rate}


class RevenueByPlan(Action):
    """Break down revenue by subscription plan"""
    
    async def _execute(self, context, **kwargs):
        plan_data = con.execute("""
            SELECT 
                plan_type,
                SUM(mrr) as total_mrr,
                COUNT(*) as customers
            FROM subscriptions
            WHERE status = 'Active'
            GROUP BY plan_type
            ORDER BY total_mrr DESC
        """).fetchall()
        
        # Build markdown table
        table = "| Plan | MRR | Customers |\n|------|-----|--------|\n"
        for plan, mrr, count in plan_data:
            table += f"| {plan} | ${mrr:,.2f} | {count} |\n"
        
        output = Markdown(f"**Revenue by Plan**\n\n{table}")
        
        return [output], {
            'plan_breakdown': {
                plan: {'mrr': mrr, 'customers': count}
                for plan, mrr, count in plan_data
            }
        }
```

## 4. Organize tasks into sections

Sections group related tasks and render them in collapsible accordions. They execute tasks sequentially and pass context between them:

```python title="report.py" linenums="100"
import panel as pn

# Enable Panel extensions
pn.extension('tabulator')

# Build the report
report = Report(
    # Section 1: Revenue Metrics
    Section(
        MRRCalculation(title="Monthly Recurring Revenue"),
        RevenueByPlan(title="Plan Distribution"),
        title="Revenue Overview"  # (1)!
    ),
    
    # Section 2: Customer Health
    Section(
        ChurnAnalysis(title="Churn Metrics"),
        title="Customer Health"
    ),
    
    title="Subscription Analytics Report"  # (2)!
)
```

1. Section titles appear in the accordion interface
2. Report title appears at the top of the dashboard

**Key concepts:**

- **Sections** group related analyses logically
- **Sequential execution** ensures data flows from task to task
- **Accordion UI** lets users focus on specific sections
- **Context accumulation** makes all previous results available

## 5. Add AI-powered insights

Combine deterministic analysis with AI agents using `ActorTask`. This wraps an AI agent so it can access context from previous tasks:

```python title="report.py" linenums="125"
from lumen.ai.agents import SQLAgent
from lumen.ai.llm import OpenAILLM
from lumen.ai.report import ActorTask

# Initialize LLM and agent
llm = OpenAILLM(model="gpt-4")
sql_agent = SQLAgent(
    llm=llm,
    tables=['customers', 'subscriptions', 'churn_log']
)

# Wrap in ActorTask
sql_analysis = ActorTask(
    sql_agent,
    title="Deep Dive Analysis",
    instruction="""
    Based on the metrics in context:
    - Total MRR: ${total_mrr:,.2f}
    - Churn Rate: {churn_rate}%
    
    Please analyze:
    1. Which customer segments have highest LTV?
    2. What patterns exist in churn timing?
    3. Generate visualizations to support your findings
    """  # (1)!
)

# Add AI section to report
report.append(Section(
    sql_analysis,
    title="AI Insights"
))
```

1. Instructions can reference values from context using `{variable}` syntax

**Key concepts:**

- **ActorTask** wraps any `Actor` (agents, tools, custom actors)
- **Context injection** allows AI to reference previous results
- **Natural language** instructions guide the AI's analysis
- **Mixed paradigm** combines deterministic + AI approaches

## 6. Run the report

Launch the interactive report interface:

```python title="report.py" linenums="160"
if __name__ == "__main__":
    report.show(port=5006)
```

Run the script:

```bash
python report.py
```

Open `http://localhost:5006` and click the **▶ Execute** button to run all sections.

### Interactive features

The report UI provides several controls:

- **▶ Execute** - Run all sections sequentially
- **🗑 Clear** - Reset all outputs
- **⬆️⬇️ Collapse/Expand** - Toggle all sections
- **⚙️ Settings** - Configure report title and sections
- **📥 Export** - Download as Jupyter notebook

## 7. Export to notebook

After execution, export the report to a Jupyter notebook for sharing or further analysis:

```python
import asyncio

# Execute the report
asyncio.run(report.execute())

# Export to notebook
notebook_string = report.to_notebook()

# Save to file
with open('subscription_report.ipynb', 'w') as f:
    f.write(notebook_string)
```

The exported notebook includes:

- All section headers as markdown cells
- Visualizations and outputs
- Preamble with required imports

## Advanced patterns

### Context schemas

Define task dependencies explicitly using schemas:

```python
from lumen.ai.actor import ContextModel
from typing_extensions import NotRequired

class RevenueInputs(ContextModel):
    """Define required context inputs"""
    customer_count: int
    
class RevenueOutputs(ContextModel):
    """Define context outputs for downstream tasks"""
    arpu: float

class ARPUCalculation(Action):
    input_schema = RevenueInputs  # (1)!
    output_schema = RevenueOutputs  # (2)!
    
    async def _execute(self, context, **kwargs):
        # Access validated inputs
        customer_count = context['customer_count']
        total_mrr = context['total_mrr']
        
        arpu = total_mrr / customer_count
        
        return [Markdown(f"ARPU: ${arpu:.2f}")], {'arpu': arpu}
```

1. `input_schema` validates required context is available
2. `output_schema` declares what this task provides

**Benefits:**

- **Validation** catches missing dependencies before execution
- **Documentation** makes task requirements explicit  
- **Type safety** ensures correct data types
- **IDE support** enables autocomplete for context keys

### Task invalidation

When you modify data in a task, downstream tasks automatically invalidate and re-execute:

```python
# Execute report
await report.execute()

# Modify a task's output
report[0][0].out_context = {'total_mrr': 75000}

# Automatically triggers re-execution of dependent tasks
await asyncio.sleep(0.1)  # Brief pause for invalidation to propagate
```

This ensures reports stay consistent when inputs change.

### Custom AI actors

Create domain-specific actors for specialized analysis:

```python
from lumen.ai.actor import Actor
from lumen.ai.prompts import Prompt

class ChurnPredictor(Actor):
    """Predict churn risk using customer behavior"""
    
    prompts = {
        'analyze': Prompt(
            system_block="""
            You are a churn prediction expert. Analyze customer
            behavior patterns to identify at-risk accounts.
            """,
            template_block="""
            Customer data: {customer_data}
            Usage patterns: {usage_data}
            
            Identify warning signs and recommend retention actions.
            """
        )
    }
    
    async def _respond(self, messages, context, **kwargs):
        # Custom prediction logic
        customer_data = context.get('customer_data')
        # ... analysis code ...
        return outputs, out_context
```

Use it in reports:

```python
churn_task = ActorTask(
    ChurnPredictor(),
    title="Churn Risk Assessment"
)
```

## Best practices

### 1. Modular design

Keep Actions focused on single responsibilities:

✅ **Good:** `MRRCalculation`, `ChurnAnalysis`, `PlanDistribution`  
❌ **Bad:** `DoEverythingAnalysis`

### 2. Rich context

Provide detailed context for downstream tasks:

```python
return [output], {
    'total_mrr': mrr,
    'active_subs': subs,
    'avg_mrr': avg,
    'mrr_growth': growth_rate,  # Add derived metrics
    'plan_breakdown': breakdown  # Include supporting data
}
```

### 3. Clear instructions

For AI tasks, be specific about what you want:

```python
instruction="""
Based on {total_mrr} MRR and {churn_rate}% churn:

1. Calculate customer lifetime value (LTV)
2. Identify at-risk segments
3. Recommend 3 retention strategies

Use SQL queries against the 'subscriptions' table.
Format output as an executive summary.
"""
```

### 4. Error handling

Actions can handle errors gracefully:

```python
async def _execute(self, context, **kwargs):
    try:
        result = con.execute(query).fetchone()
    except Exception as e:
        error_msg = Markdown(f"⚠️ Analysis failed: {e}")
        return [error_msg], {'error': str(e)}
```

### 5. Progressive disclosure

Start with high-level sections, add detail in nested tasks:

```python
Report(
    Section(
        HighLevelMetrics(),
        title="Executive Summary"
    ),
    Section(
        DetailedBreakdown(),
        TrendAnalysis(),
        CohortAnalysis(),
        title="Deep Dive"
    )
)
```

## Common patterns

### Metric calculation

Standard pattern for computing business metrics:

```python
class MetricAction(Action):
    async def _execute(self, context, **kwargs):
        # 1. Query data
        data = con.execute(query).fetchall()
        
        # 2. Calculate metric
        metric_value = compute_metric(data)
        
        # 3. Format output
        output = Markdown(f"Metric: {metric_value}")
        
        # 4. Return with context
        return [output], {'metric': metric_value}
```

### Comparative analysis

Compare metrics across dimensions:

```python
class ComparisonAction(Action):
    async def _execute(self, context, **kwargs):
        # Get baseline from context
        baseline = context['baseline_value']
        
        # Calculate current
        current = con.execute(query).fetchone()[0]
        
        # Compute change
        change = ((current - baseline) / baseline) * 100
        
        output = Markdown(f"""
        Current: {current}
        Change: {change:+.1f}%
        """)
        
        return [output], {'current': current, 'change': change}
```

### Time series

Analyze trends over time:

```python
class TrendAction(Action):
    async def _execute(self, context, **kwargs):
        # Get monthly data
        data = con.execute("""
            SELECT 
                DATE_TRUNC('month', date) as month,
                SUM(value) as total
            FROM table
            GROUP BY month
            ORDER BY month
        """).fetchdf()
        
        # Create visualization
        chart = data.hvplot.line(x='month', y='total')
        
        return [chart], {'trend_data': data.to_dict()}
```

## Troubleshooting

### Database connection issues

```python
# Use context manager for safe connections
async def _execute(self, context, **kwargs):
    with duckdb.connect('db.duckdb') as con:
        result = con.execute(query).fetchone()
```

### Context not available

Check task execution order and schemas:

```python
# Validate before execution
issues, types = report.validate()
if issues:
    for issue in issues:
        print(f"Error: {issue}")
```

### AI tasks failing

Verify API keys and check LLM configuration:

```python
# Test LLM directly
from lumen.ai.llm import OpenAILLM

llm = OpenAILLM()
response = await llm.invoke("Test message")
print(response)
```

## Next steps

- Explore [Agents](../../reference/api.md#agents) for more AI capabilities
- Learn about [Custom Analyses](../../configuration/analyses.md)
- Build [Interactive Dashboards](penguins_dashboard_spec.md)
- Check the [API Reference](../../reference/api.md)

## Resources

- [Lumen GitHub Repository](https://github.com/holoviz/lumen)
- [Panel Documentation](https://panel.holoviz.org)
- [DuckDB SQL Reference](https://duckdb.org/docs/sql/introduction)
- [Example Reports](https://github.com/holoviz/lumen/tree/main/examples)
