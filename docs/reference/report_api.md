# Report API Reference

Quick reference for building reports with Lumen.

**See also:** [Building AI-Powered Reports Tutorial](../../examples/tutorials/building_reports.md) — Step-by-step guide with complete working examples.

## Core Classes

### Report

Top-level container for organizing sections.

```python
from lumen.ai.report import Report

report = Report(
    section1,
    section2,
    title="My Report"
)

# Or pass as list
report = Report(
    tasks=[section1, section2],
    title="My Report"
)
```

**Key methods:**

- `execute(context=None, **kwargs)` - Run all sections
- `show(port=5006)` - Launch interactive UI
- `to_notebook()` - Export to Jupyter notebook format
- `append(section)` - Add a section
- `reset(start=0)` - Clear outputs
- `validate()` - Check task dependencies

**Key attributes:**

- `title` - Report title shown in UI
- `status` - Current execution state: 'idle', 'running', 'success', 'error'
- `views` - List of all rendered outputs
- `out_context` - Merged context from all tasks

### Section

Groups related tasks in collapsible accordion.

```python
from lumen.ai.report import Section

section = Section(
    task1,
    task2,
    task3,
    title="Section Name"
)
```

**Key methods:**

- `execute(context, **kwargs)` - Run all tasks in sequence
- `append(task)` - Add a task
- `remove(task)` - Remove a task
- `invalidate(keys, start=0)` - Mark tasks as stale
- `merge(other_section)` - Combine with another section

**Key attributes:**

- `title` - Section heading
- `abort_on_error` - Stop on first error (default: True)
- `level` - Heading level (default: 2)

## Task Types

### Action

Custom deterministic analysis task.

```python
from lumen.ai.report import Action
from panel.pane import Markdown

class MyAnalysis(Action):
    """Docstring describes what this does"""
    
    async def _execute(self, context, **kwargs):
        # 1. Perform analysis
        result = analyze_data(context)
        
        # 2. Create output
        output = Markdown(f"Result: {result}")
        
        # 3. Return (outputs, context)
        return [output], {'result': result}
```

**Key attributes:**

- `title` - Task heading
- `instruction` - Optional text instruction
- `context` - Initial context dict
- `history` - Conversation history (for AI tasks)
- `status` - Current state
- `views` - Rendered outputs
- `out_context` - Generated context
- `render_outputs` - Whether to display outputs (default: True)

**Context schemas:**

```python
from lumen.ai.actor import ContextModel

class MyInputs(ContextModel):
    """Required inputs"""
    total_revenue: float
    customer_count: int

class MyOutputs(ContextModel):
    """Provided outputs"""
    arpu: float

class MyAction(Action):
    input_schema = MyInputs
    output_schema = MyOutputs
    
    async def _execute(self, context, **kwargs):
        arpu = context['total_revenue'] / context['customer_count']
        return [], {'arpu': arpu}
```

### ActorTask

Wraps an AI actor (agent, tool, etc.) for use in reports.

```python
from lumen.ai.report import ActorTask
from lumen.ai.agents import SQLAgent
from lumen.ai.llm import OpenAILLM

agent = SQLAgent(
    llm=OpenAILLM(),
    tables=['customers', 'orders']
)

task = ActorTask(
    agent,
    title="SQL Analysis",
    instruction="""
    Analyze customer purchase patterns.
    Focus on: {metric_name}
    """
)
```

**Context injection:**

Instructions support `{variable}` syntax to reference context:

```python
task = ActorTask(
    agent,
    instruction="Total MRR is ${total_mrr:,.2f}. Analyze trends."
)
```

## Context System

### Context Flow

```python
# Task 1 provides context
class Task1(Action):
    async def _execute(self, context, **kwargs):
        return [], {'value_a': 100}

# Task 2 consumes context
class Task2(Action):
    async def _execute(self, context, **kwargs):
        a = context['value_a']  # Access previous result
        return [], {'value_b': a * 2}

# Task 3 has access to both
class Task3(Action):
    async def _execute(self, context, **kwargs):
        a = context['value_a']  # From Task1
        b = context['value_b']  # From Task2
        return [], {'value_c': a + b}
```

### Context Merging

Within a section, contexts merge using Last-Write-Wins (LWW):

```python
section = Section(
    TaskA(),  # Returns {'x': 1, 'y': 2}
    TaskB(),  # Returns {'x': 3, 'z': 4}
    TaskC()   # Sees {'x': 3, 'y': 2, 'z': 4}
)
```

### Initial Context

Provide context to tasks:

```python
# At report level
report = Report(
    section,
    context={'database': 'prod.db'}
)

# At section level
section = Section(
    task,
    context={'start_date': '2024-01-01'}
)

# At task level
task = MyAction(
    context={'filter': 'active'}
)

# At execution
await report.execute(context={'override': True})
```

## Visualization

### Output Types

Reports can render various Panel objects:

```python
from panel.pane import Markdown, HTML
from panel.widgets import Tabulator
import hvplot.pandas

async def _execute(self, context, **kwargs):
    outputs = [
        Markdown("# Header"),           # Markdown text
        HTML("<div>Custom HTML</div>"), # HTML content
        df.hvplot.line(),               # HoloViews plot
        Tabulator(df),                  # Data table
    ]
    return outputs, {}
```

### Lumen Outputs

For Lumen-specific visualizations:

```python
from lumen.ai.views import LumenOutput, SQLOutput
from lumen.views import Panel

# Wrap Panel components
view = Panel(object=my_panel_object, pipeline=pipeline)
output = LumenOutput(component=view, title="My View")

# SQL-specific output
sql_output = SQLOutput(
    query="SELECT * FROM table",
    result=df,
    title="Query Results"
)
```

## Advanced Features

### Task Invalidation

When context changes, dependent tasks automatically invalidate:

```python
# Execute report
await report.execute()

# Modify a task's output
report[0][1].out_context = {'new_value': 200}

# Dependent tasks automatically marked for re-execution
# Call execute() to re-run invalidated tasks
await report.execute()
```

### Programmatic Control

```python
# Access tasks
section = report[0]        # First section
task = section[1]          # Second task in section
task = report[0][1]        # Shorthand

# Check status
if task.status == 'success':
    print(task.out_context)

# Modify tasks
report.append(Section(new_task))
section.insert(0, new_task)
section.remove(old_task)

# Merge sections
section1.merge(section2)
```

### Validation

Validate task dependencies before execution:

```python
# Returns (issues, types) tuple
issues, output_types = report.validate()

if issues:
    for issue in issues:
        print(f"{issue.path}: {issue.message}")
else:
    await report.execute()
```

**Validation checks:**

- Required inputs are available
- No circular dependencies  
- Output types match expected types
- No mutual exclusions violated

### Notebook Export

```python
# Execute first
await report.execute()

# Get notebook JSON string
nb_json = report.to_notebook()

# Save to file
with open('report.ipynb', 'w') as f:
    f.write(nb_json)

# Export includes:
# - Import preamble
# - Section headers as markdown
# - Task outputs as cells
```

## Error Handling

### Abort on Error

```python
# Stop section on first error (default)
section = Section(
    task1,
    task2,
    abort_on_error=True
)

# Continue despite errors
section = Section(
    task1,
    task2,
    abort_on_error=False
)
```

### Error Context

Failed tasks provide error context:

```python
async def _execute(self, context, **kwargs):
    try:
        result = risky_operation()
    except Exception as e:
        # Error stored in context
        return [], {
            '__error__': str(e),
            '__error_type__': type(e)
        }
```

### Status Tracking

Monitor execution state:

```python
# Task status values
task.status  # 'idle', 'running', 'success', 'error'

# Section inherits status
section.status  # 'error' if any task failed

# Report inherits status  
report.status  # Overall execution state
```

## UI Customization

### Editor Views

Customize task configuration UI:

```python
class MyAction(Action):
    param1 = param.String(default="value")
    param2 = param.Integer(default=10)
    
    def editor(self, show_title=True):
        from panel.widgets import TextInput, IntInput
        
        return Column(
            TextInput.from_param(self.param.param1),
            IntInput.from_param(self.param.param2)
        )
```

### Styling

Control appearance:

```python
from panel.layout import Column

class StyledAction(Action):
    def _init_view(self):
        self._view = Column(
            sizing_mode='stretch_width',
            styles={'background': '#f0f0f0', 'padding': '10px'}
        )
```

## Best Practices

### 1. Clear naming

Use descriptive titles and class names:

```python
# Good
class CustomerLifetimeValue(Action):
    async def _execute(self, context, **kwargs):
        ...

task = CustomerLifetimeValue(title="LTV Analysis")

# Bad  
class Action1(Action):
    ...
```

### 2. Rich context

Provide useful data for downstream tasks:

```python
return [output], {
    'total_revenue': 50000,
    'customer_count': 100,
    'arpu': 500,
    'growth_rate': 0.15,
    'breakdown': {'basic': 20, 'pro': 60, 'enterprise': 20}
}
```

### 3. Modular tasks

Keep Actions focused:

```python
# Good - focused tasks
class CalculateMRR(Action): ...
class CalculateChurn(Action): ...
class CalculateLTV(Action): ...

# Bad - does too much
class CalculateEverything(Action): ...
```

### 4. Document schemas

Use schemas for clarity:

```python
class Inputs(ContextModel):
    """Data required from previous tasks"""
    total_mrr: float
    active_customers: int

class Outputs(ContextModel):  
    """Data provided to downstream tasks"""
    arpu: float
    ltv: float
```

### 5. Test independently

Test tasks before adding to reports:

```python
import asyncio

task = MyAction()
outputs, context = asyncio.run(task.execute({
    'required_input': 100
}))

assert 'expected_output' in context
```

## Common Patterns

### Pipeline pattern

Sequential analysis with data flowing through tasks:

```python
Section(
    LoadData(title="1. Load"),
    TransformData(title="2. Transform"),  
    AnalyzeData(title="3. Analyze"),
    VisualizeData(title="4. Visualize"),
    title="Analysis Pipeline"
)
```

### Parallel sections

Independent analyses in separate sections:

```python
Report(
    Section(RevenueAnalysis(), title="Revenue"),
    Section(ChurnAnalysis(), title="Churn"),
    Section(GrowthAnalysis(), title="Growth"),
    title="Business Metrics"
)
```

### AI + deterministic

Combine both approaches:

```python
Section(
    CalculateMetrics(),      # Deterministic
    ActorTask(sql_agent),    # AI-powered
    SummarizeFindings(),     # Deterministic
    title="Hybrid Analysis"
)
```

### Iterative refinement

Build reports incrementally:

```python
report = Report(title="Analysis")

# Start simple
report.append(Section(BasicMetrics(), title="Metrics"))

# Add detail
report[0].append(DetailedBreakdown())

# Add AI
report.append(Section(
    ActorTask(agent),
    title="Insights"
))
```

## Examples

See the [Building Reports Tutorial](../tutorials/building_reports.md) for complete examples.
