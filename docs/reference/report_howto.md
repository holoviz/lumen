# Report How-To Guides

Quick recipes for common report tasks.

## Quick Start

### Minimal report

```python
from lumen.ai.report import Action, Report, Section
from panel.pane import Markdown
import panel as pn

pn.extension()

class HelloWorld(Action):
    async def _execute(self, context, **kwargs):
        return [Markdown("**Hello, World!**")], {}

report = Report(
    Section(HelloWorld(), title="Greeting"),
    title="My First Report"
)

report.show()
```

## Working with Data

### Connect to database

```python
import duckdb

con = duckdb.connect('data.db')

class QueryAction(Action):
    async def _execute(self, context, **kwargs):
        df = con.execute("SELECT * FROM table").fetchdf()
        return [Markdown(f"Rows: {len(df)}")], {'data': df}
```

### Query with parameters

```python
class ParameterizedQuery(Action):
    async def _execute(self, context, **kwargs):
        # Get filter from context
        status = context.get('status_filter', 'Active')
        
        df = con.execute(
            "SELECT * FROM customers WHERE status = ?",
            [status]
        ).fetchdf()
        
        return [Markdown(f"Found {len(df)} customers")], {'filtered_data': df}
```

### Load from CSV

```python
import pandas as pd

class LoadCSV(Action):
    async def _execute(self, context, **kwargs):
        df = pd.read_csv('data.csv')
        
        summary = Markdown(f"""
        **Data Loaded**
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        """)
        
        return [summary], {'raw_data': df}
```

## Creating Visualizations

### Simple chart

```python
import hvplot.pandas

class ChartAction(Action):
    async def _execute(self, context, **kwargs):
        df = context['data']
        chart = df.hvplot.line(x='date', y='value')
        return [chart], {}
```

### Multiple visualizations

```python
class DashboardAction(Action):
    async def _execute(self, context, **kwargs):
        df = context['data']
        
        # Create multiple charts
        line_chart = df.hvplot.line(x='date', y='value')
        bar_chart = df.hvplot.bar(x='category', y='count')
        table = df.head(10).pipe(Tabulator)
        
        return [line_chart, bar_chart, table], {}
```

### Custom Panel layout

```python
from panel.layout import Row, Column

class CustomLayout(Action):
    async def _execute(self, context, **kwargs):
        df = context['data']
        
        chart1 = df.hvplot.line()
        chart2 = df.hvplot.bar()
        
        layout = Row(
            Column(chart1, sizing_mode='stretch_both'),
            Column(chart2, sizing_mode='stretch_both')
        )
        
        return [layout], {}
```

## Context Management

### Pass data between tasks

```python
class Task1(Action):
    async def _execute(self, context, **kwargs):
        data = {'metric': 100}
        return [Markdown("Task 1 done")], data

class Task2(Action):
    async def _execute(self, context, **kwargs):
        # Access from Task1
        metric = context['metric']
        result = metric * 2
        return [Markdown(f"Result: {result}")], {'result': result}

section = Section(Task1(), Task2())
```

### Conditional execution

```python
class ConditionalTask(Action):
    async def _execute(self, context, **kwargs):
        threshold = context.get('threshold', 100)
        
        if context['value'] > threshold:
            message = "⚠️ Value exceeds threshold!"
        else:
            message = "✓ Value within limits"
        
        return [Markdown(message)], {'alert': context['value'] > threshold}
```

### Transform context data

```python
class Transform(Action):
    async def _execute(self, context, **kwargs):
        raw_data = context['raw_data']
        
        # Transform
        processed = raw_data.groupby('category').sum()
        
        return [], {
            'raw_data': raw_data,        # Keep original
            'processed_data': processed   # Add processed
        }
```

## Using AI

### Basic AI task

```python
from lumen.ai.agents import SQLAgent
from lumen.ai.llm import OpenAILLM
from lumen.ai.report import ActorTask

agent = SQLAgent(
    llm=OpenAILLM(),
    tables=['customers']
)

ai_task = ActorTask(
    agent,
    title="AI Analysis",
    instruction="Find top customers by revenue"
)
```

### AI with context

```python
ai_task = ActorTask(
    agent,
    instruction="""
    Total revenue is ${total_revenue:,.2f}
    Average customer value is ${avg_customer_value:.2f}
    
    Analyze customer segmentation and recommend
    strategies to increase these metrics.
    """
)
```

### Custom AI actor

```python
from lumen.ai.actor import Actor
from lumen.ai.prompts import Prompt

class CustomActor(Actor):
    prompts = {
        'analyze': Prompt(
            system_block="You are an expert analyst.",
            template_block="Analyze this: {data}"
        )
    }
    
    async def _respond(self, messages, context, **kwargs):
        # Your custom logic
        result = "Analysis result"
        return [Markdown(result)], {'analysis': result}

task = ActorTask(CustomActor(), title="Custom Analysis")
```

## Report Structure

### Multiple sections

```python
report = Report(
    Section(
        MetricA(),
        MetricB(),
        title="Overview"
    ),
    Section(
        DetailedAnalysisA(),
        DetailedAnalysisB(),
        title="Deep Dive"
    ),
    Section(
        SummaryTask(),
        title="Summary"
    ),
    title="Complete Report"
)
```

### Nested sections

```python
section = Section(
    TaskGroup(
        SubTask1(),
        SubTask2(),
        title="Sub-analysis"
    ),
    MainTask(),
    title="Main Section"
)
```

### Dynamic sections

```python
# Build sections programmatically
sections = []

for category in ['revenue', 'churn', 'growth']:
    section = Section(
        AnalyzeCategory(category=category),
        title=f"{category.title()} Analysis"
    )
    sections.append(section)

report = Report(*sections, title="Multi-Category Report")
```

## Error Handling

### Graceful failure

```python
class SafeAction(Action):
    async def _execute(self, context, **kwargs):
        try:
            result = risky_operation()
            return [Markdown(f"Success: {result}")], {'result': result}
        except Exception as e:
            error_msg = Markdown(f"⚠️ Error: {str(e)}")
            return [error_msg], {'error': str(e)}
```

### Continue on error

```python
# Section continues even if a task fails
section = Section(
    Task1(),
    Task2(),
    Task3(),
    abort_on_error=False  # Keep going
)
```

### Validate before execution

```python
# Check dependencies first
issues, types = report.validate()

if issues:
    print("Validation failed:")
    for issue in issues:
        print(f"  - {issue.path}: {issue.message}")
else:
    await report.execute()
```

## Exporting

### Export to notebook

```python
import asyncio

# Execute
asyncio.run(report.execute())

# Export
nb_json = report.to_notebook()

# Save
with open('report.ipynb', 'w') as f:
    f.write(nb_json)
```

### Export specific section

```python
# Execute just one section
section = report[0]
await section.execute()

# Export it
nb_json = section.to_notebook()
```

### Custom export

```python
class ExportAction(Action):
    async def _execute(self, context, **kwargs):
        # Collect all context
        export_data = {
            'metrics': context,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        with open('results.json', 'w') as f:
            json.dump(export_data, f)
        
        return [Markdown("✓ Exported to results.json")], {}
```

## Performance

### Lazy loading

```python
class LazyLoad(Action):
    async def _execute(self, context, **kwargs):
        # Only load if needed
        if 'large_dataset' not in context:
            df = load_large_dataset()
            context['large_dataset'] = df
        
        # Use cached data
        df = context['large_dataset']
        result = df.head()
        
        return [Tabulator(result)], {}
```

### Batch operations

```python
class BatchAction(Action):
    async def _execute(self, context, **kwargs):
        # Process in batches to avoid memory issues
        df = context['data']
        batch_size = 1000
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            result = process_batch(batch)
            results.append(result)
        
        final = pd.concat(results)
        return [Tabulator(final.head())], {'processed': final}
```

### Parallel execution

```python
import asyncio

class ParallelAction(Action):
    async def _execute(self, context, **kwargs):
        # Run multiple queries in parallel
        tasks = [
            asyncio.create_task(query_a()),
            asyncio.create_task(query_b()),
            asyncio.create_task(query_c())
        ]
        
        results = await asyncio.gather(*tasks)
        
        return [Markdown("All queries complete")], {
            'result_a': results[0],
            'result_b': results[1],
            'result_c': results[2]
        }
```

## Customization

### Custom styling

```python
class StyledAction(Action):
    async def _execute(self, context, **kwargs):
        from panel.pane import HTML
        
        styled = HTML("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        ">
            <h2>Custom Styled Output</h2>
            <p>Your content here</p>
        </div>
        """)
        
        return [styled], {}
```

### Interactive widgets

```python
from panel.widgets import Select, Button
from panel.layout import Column

class InteractiveAction(Action):
    async def _execute(self, context, **kwargs):
        df = context['data']
        
        # Create widgets
        column_select = Select(
            options=list(df.columns),
            value=df.columns[0],
            name='Select Column'
        )
        
        # Create reactive chart
        chart = pn.bind(
            lambda col: df.hvplot.hist(y=col),
            column_select
        )
        
        layout = Column(column_select, chart)
        
        return [layout], {}
```

### Progress indicators

```python
class ProgressAction(Action):
    async def _execute(self, context, **kwargs):
        from panel.widgets import Progress
        import asyncio
        
        progress = Progress(value=0, max=100)
        self._view.append(progress)
        
        for i in range(0, 101, 10):
            await asyncio.sleep(0.1)
            progress.value = i
        
        return [Markdown("✓ Complete")], {}
```

## Testing

### Test individual action

```python
import asyncio

def test_action():
    action = MyAction()
    outputs, context = asyncio.run(action.execute({
        'input_data': test_data
    }))
    
    assert len(outputs) > 0
    assert 'expected_key' in context
```

### Test section

```python
def test_section():
    section = Section(Task1(), Task2())
    outputs, context = asyncio.run(section.execute())
    
    assert section.status == 'success'
    assert 'result' in context
```

### Mock data

```python
class MockDataAction(Action):
    async def _execute(self, context, **kwargs):
        # Use mock data for testing
        if context.get('testing'):
            df = create_mock_data()
        else:
            df = load_real_data()
        
        return [Tabulator(df)], {'data': df}

# Test with mock
await action.execute({'testing': True})
```

## Debugging

### Print context

```python
class DebugAction(Action):
    async def _execute(self, context, **kwargs):
        # See what's available
        debug_info = Markdown(f"""
        **Context Keys:**
        {', '.join(context.keys())}
        
        **Context Values:**
        ```python
        {pprint.pformat(context)}
        ```
        """)
        
        return [debug_info], {}
```

### Log execution

```python
import logging

logger = logging.getLogger(__name__)

class LoggingAction(Action):
    async def _execute(self, context, **kwargs):
        logger.info(f"Executing with context keys: {context.keys()}")
        
        result = perform_analysis()
        
        logger.info(f"Generated result: {result}")
        
        return [Markdown(str(result))], {'result': result}
```

### Step through

```python
class StepDebugAction(Action):
    async def _execute(self, context, **kwargs):
        # Add breakpoint for debugging
        import pdb; pdb.set_trace()
        
        result = complex_operation(context)
        
        return [Markdown(result)], {'result': result}
```

## Common Issues

### Context key not found

```python
# Problem: KeyError when accessing context
value = context['key']  # ❌ Fails if key missing

# Solution: Use .get() with default
value = context.get('key', default_value)  # ✓ Safe
```

### Outputs not rendering

```python
# Problem: Wrong output type
return ["string"], {}  # ❌ Strings not rendered

# Solution: Wrap in Panel object
from panel.pane import Markdown
return [Markdown("string")], {}  # ✓ Renders correctly
```

### Section not updating

```python
# Problem: Forgot to await
report.execute()  # ❌ Returns coroutine

# Solution: Use await or asyncio.run
await report.execute()  # ✓ In async context
asyncio.run(report.execute())  # ✓ In sync context
```

### AI task failing

```python
# Problem: Missing API key
agent = SQLAgent(llm=OpenAILLM())  # ❌ No key set

# Solution: Set environment variable first
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
agent = SQLAgent(llm=OpenAILLM())  # ✓ Works
```

## More Examples

See the [Building Reports Tutorial](../tutorials/building_reports.md) for complete working examples.
