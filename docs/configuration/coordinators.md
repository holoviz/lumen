# :material-sitemap: Coordinators

**Coordinators decide which agents answer your questions and in what order.**

Most users never need to change the coordinator. The default works well for typical data exploration.

## Should you customize the coordinator?

**Probably not.** The default Planner handles most cases well.

Consider customizing only if:

- Planning is too slow for simple queries (try DependencyResolver)
- You want to see detailed planning steps (enable verbose mode)
- Plans include too many unnecessary steps

## Change coordinators

### Use DependencyResolver for faster planning

The DependencyResolver skips comprehensive planning and jumps straight to execution:

``` py title="Use DependencyResolver"
import lumen.ai as lmai
from lumen.ai.coordinator import DependencyResolver

ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator=DependencyResolver
)
ui.servable()
```

**When to use DependencyResolver:**

- Simple, single-purpose queries ("Show me a chart of X")
- When planning takes longer than execution
- When you don't need automatic validation

**When to keep the default Planner:**

- Complex multi-step analysis
- When you want to see the full plan before execution
- When automatic validation is valuable

## Adjust planning behavior

### Show detailed planning steps

See exactly how Lumen creates and executes plans:

``` py title="Enable verbose mode" hl_lines="3"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'verbose': True}
)
ui.servable()
```

This shows which agents are considered, why each is selected, and what data flows between steps.

### Disable automatic validation

Skip the validation step to get results faster:

``` py title="Disable validation" hl_lines="3"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'validation_enabled': False}
)
ui.servable()
```

!!! warning "Only disable validation if you trust the results"
    Validation catches errors and confirms queries were answered correctly. Only disable during rapid development iteration.

### Control pre-planning data lookup

The Planner looks up table information before creating a plan. Disable this if you already know what data is available:

``` py title="Skip table lookup" hl_lines="3"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'planner_tools': []}
)
ui.servable()
```

## How coordinators work

### Planner creates a complete plan upfront

The Planner is Lumen's default coordinator:

``` mermaid
graph TB
    A[User Query] --> B[Gather Context]
    B --> C[Create Plan]
    C --> D[Show Checklist]
    D --> E[Execute Step 1]
    E --> F[Execute Step 2]
    F --> G[Execute Step N]
    G --> H[Validate Results]
```

**Example:** "Show me average bill length by species as a bar chart"

The Planner creates this checklist:
```
☐ Find tables with penguin data
☐ Query data and calculate averages by species
☐ Create a bar chart
☐ Summarize the findings
☐ Verify the query was answered
```

Then executes each step in order, checking them off as it goes.

### DependencyResolver works backward from the goal

The DependencyResolver picks the final agent first, then figures out what it needs:

``` mermaid
graph BT
    A[TableLookup] --> B[SQLAgent]
    B --> C[VegaLiteAgent]
    C --> D[Goal Achieved]
```

**Same example:** "Show me average bill length by species as a bar chart"

DependencyResolver thinks:
```
Goal: VegaLiteAgent (needs pipeline)
  ↑ SQLAgent (needs metaset)
    ↑ TableLookup (no dependencies)

Execution: TableLookup → SQLAgent → VegaLiteAgent
```

No planning phase, just direct execution.

## Common issues

### "Planner failed to come up with viable plan"

The planner couldn't create a valid plan after multiple attempts.

**How to fix:**

1. Rephrase your question more specifically
2. Check that required data sources are loaded
3. Try with `verbose=True` to see what went wrong
4. Use DependencyResolver for simpler queries

### Plans include too many steps

The planner is being overly cautious.

**How to fix:**

1. Ask follow-up questions to reuse existing data
2. Disable validation: `coordinator_params={'validation_enabled': False}`
3. Use DependencyResolver for straightforward queries

### Planning takes too long

The Planner is gathering too much context upfront.

**How to fix:**

1. Switch to DependencyResolver
2. Disable pre-planning lookup: `coordinator_params={'planner_tools': []}`

### Agents run in wrong order

Dependencies weren't resolved correctly.

**How to fix:**

1. Enable verbose mode to see the dependency chain
2. Report the issue on GitHub with a reproducible example

## Coordinator comparison

| Feature | Planner (default) | DependencyResolver |
|---------|------------------|-------------------|
| **Planning speed** | Slower (creates plan first) | Faster (no planning phase) |
| **Best for** | Complex queries | Simple queries |
| **Shows plan** | Yes, with checklist | No, executes directly |
| **Validation** | Automatic | Not included |
| **LLM calls** | More (comprehensive) | Fewer (minimal) |
| **Recommended for** | Most use cases | Speed-critical simple queries |

## When to create a custom coordinator

**Almost never.** The built-in coordinators handle nearly all use cases.

Only create a custom coordinator if:

- You have very specific orchestration requirements
- You're building a specialized application with unique workflows
- You've exhausted all configuration options

See the [Contributing guide](../reference/contributing.md) if you think Lumen needs a new coordinator type.
