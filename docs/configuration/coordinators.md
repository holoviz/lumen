# Coordinators

**Coordinators decide which agents answer your questions and in what order.**

Most users never need to change the coordinator. The default works well for typical data exploration.

## Skip to

- [Change coordinators](#change-coordinators) - Switch to a different coordinator
- [Adjust planning](#adjust-planning-behavior) - Control how plans are created
- [Troubleshooting](#common-issues) - Fix planning problems

## Should you customize the coordinator?

**No, probably not.** The default Planner handles most cases well.

Consider customizing only if:

- Planning is too slow for simple queries (try DependencyResolver)
- You want to see detailed planning steps (enable verbose mode)
- Plans include too many unnecessary steps

## Change coordinators

### Use DependencyResolver for faster planning

The DependencyResolver skips comprehensive planning and jumps straight to execution. This makes simple queries faster.

```python
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

```python
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'verbose': True}
)
ui.servable()
```

This shows:

- Which agents are considered
- Why each agent is selected
- What data flows between steps

Useful for understanding Lumen's decisions or debugging unexpected behavior.

### Include more conversation history

By default, Lumen remembers your last 3 messages when planning. Increase this if you're having multi-turn conversations:

```python
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'history': 5}  # Remember last 5 exchanges
)
ui.servable()
```

**Trade-offs:**

- More history = better context for follow-up questions
- More history = higher LLM costs (more tokens)

### Disable automatic validation

Skip the validation step to get results faster:

```python
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'validation_enabled': False}
)
ui.servable()
```

Only disable validation if:

- You trust the results without verification
- Speed matters more than accuracy
- You're iterating rapidly during development

### Control pre-planning data lookup

The Planner looks up table information before creating a plan. Disable this if you already have data loaded:

```python
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={'planner_tools': []}  # Skip table lookup
)
ui.servable()
```

## How coordinators work

### Planner creates a complete plan upfront

The Planner is Lumen's default coordinator. It works like this:

1. **Gathers context** - Looks up available tables and their columns
2. **Creates a plan** - Decides which agents to use and in what order
3. **Shows the plan** - Displays a checklist of steps
4. **Executes steps** - Runs agents one by one
5. **Validates results** - Checks if the query was fully answered

**Example:** You ask "Show me average bill length by species as a bar chart"

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

1. **Picks the goal agent** - "VegaLiteAgent creates charts"
2. **Checks requirements** - "VegaLiteAgent needs data"
3. **Finds providers** - "SQLAgent provides data"
4. **Repeats** - "SQLAgent needs table info"
5. **Executes forward** - Runs agents in dependency order

**Same example:** "Show me average bill length by species as a bar chart"

DependencyResolver thinks:
```
Goal: VegaLiteAgent (needs data)
  ↑ SQLAgent (needs table info)
    ↑ TableLookup (no dependencies)

Execution: TableLookup → SQLAgent → VegaLiteAgent
```

No planning phase, just direct execution.

## Common issues

### "Planner failed to come up with viable plan"

**What happened:** The planner couldn't create a valid plan after multiple attempts.

**How to fix:**

1. Rephrase your question more specifically
2. Check that required data sources are loaded
3. Try with `verbose=True` to see what went wrong
4. Use DependencyResolver for simpler queries

### Plans include too many steps

**What happened:** The planner is being overly cautious.

**How to fix:**

1. Ask follow-up questions to reuse existing data
2. Disable validation: `coordinator_params={'validation_enabled': False}`
3. Use DependencyResolver for straightforward queries

### Planning takes too long

**What happened:** The Planner is gathering too much context upfront.

**How to fix:**

1. Switch to DependencyResolver
2. Disable pre-planning lookup: `coordinator_params={'planner_tools': []}`
3. Reduce history: `coordinator_params={'history': 1}`

### Agents run in wrong order

**What happened:** Dependencies weren't resolved correctly.

**How to fix:**

1. Make sure you're using the latest version of Lumen
2. Enable verbose mode to see the dependency chain
3. Report the issue on GitHub with a reproducible example

## Coordinator comparison

| Feature | Planner (default) | DependencyResolver |
|---------|------------------|-------------------|
| **Planning speed** | Slower (full plan first) | Faster (no planning) |
| **Best for** | Complex queries | Simple queries |
| **Shows plan** | Yes, with checklist | No, executes directly |
| **Validation** | Automatic | Not included |
| **LLM calls** | More (comprehensive) | Fewer (minimal) |
| **Recommended** | Most cases | Speed-critical simple queries |

## When to create a custom coordinator

**Almost never.** The built-in coordinators handle nearly all use cases.

Only create a custom coordinator if:

- You have very specific orchestration requirements
- You're building a specialized application with unique workflows
- You've exhausted all configuration options

See the [Contributing guide](../contributing.md) if you think Lumen needs a new coordinator type.
