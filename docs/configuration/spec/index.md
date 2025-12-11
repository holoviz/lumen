# :material-file-code: Lumen YAML Specifications

Build powerful data dashboards using YAML configuration files—no Python coding required.

!!! info "About Lumen specs and Lumen AI"
    **Lumen AI runs on Lumen specs under the hood.** When you use Lumen AI to create dashboards, it generates these YAML specifications automatically. Reports created with Lumen AI are fully reproducible using the generated specs.
    
    **Historically, users wrote these specs manually** before LLMs became capable. Now, Lumen AI can generate them for you through natural language conversation. However, understanding specs is still valuable for:
    
    - **Customizing AI-generated dashboards** beyond what the AI created
    - **Building dashboards programmatically** without the AI interface
    - **Version control and reproducibility** of your dashboard configurations
    - **Advanced features** not yet supported by the AI
    
    **Choose your approach:**
    
    - Want AI to build it for you? → [Use Lumen AI](../../getting_started/using_lumen_ai.md)
    - Want to learn the underlying system? → Continue reading this guide
    - Want to customize an AI-generated dashboard? → Learn specs, then edit the YAML

## What are Lumen specs?

Lumen specifications let you build interactive data dashboards by writing YAML configuration instead of code. Define your data sources, transformations, and visualizations declaratively, then deploy with a single command.

## Why use specs?

| Benefit | Description |
|---------|-------------|
| **No coding required** | Build dashboards using simple YAML syntax |
| **Fast iteration** | Edit YAML, refresh browser—see changes instantly |
| **Version control friendly** | Track dashboard changes like any text file |
| **Reproducible** | Share exact dashboard configuration with others |
| **Extensible** | Add Python when you need custom behavior |
| **AI-transparent** | Understand and modify what Lumen AI generates |

## Quick navigation

Choose your path based on experience level:

### 🎯 New to Lumen specs?

Start with the tutorial to build your first dashboard in 15 minutes:

**[Build Dashboard with Spec](../../tutorials/build_dashboard_with_spec.md)** - Complete hands-on tutorial

Then understand the concepts:

1. **[Core Concepts](concepts.md)** - Understand how Lumen specs work
2. **[Loading Data](sources.md)** - Connect to your data sources

### 💪 Building dashboards?

Guides for common dashboard tasks:

| Task | Guide |
|------|-------|
| Load data from files | [Loading Data](sources.md) |
| Filter and transform data | [Transforming Data](pipelines.md) |
| Create plots and tables | [Visualizing Data](views.md) |
| Use variables and references | [Variables & References](variables.md) |
| Enable data downloads | [Data Downloads](downloads.md) |

### 🚀 Advanced features?

Extend Lumen with custom functionality:

- **[Custom Components](customization.md)** - Build custom sources, transforms, filters, and views
- **[Python API](python-api.md)** - Build dashboards programmatically
- **[Authentication](authentication.md)** - Secure your dashboards

### 🛠️ Ready to deploy?

Launch your dashboard:

- **[Deployment](deployment.md)** - Serve dashboards and validate configurations

## How Lumen works

Lumen follows a simple three-stage workflow:

```
┌──────────┐     ┌───────────┐     ┌────────┐
│ Sources  │ ──> │ Pipelines │ ──> │ Views  │
└──────────┘     └───────────┘     └────────┘
    │                 │                  │
    │                 │                  │
Load data      Filter/transform     Visualize
```

1. **Sources** load data from files, databases, or APIs
2. **Pipelines** filter and transform the data
3. **Views** display the results as plots, tables, or other visualizations

## Example dashboard

Here's a complete dashboard specification:

```yaml
config:
  title: Sales Dashboard
  theme: dark

sources:
  sales_data:
    type: file
    tables:
      sales: data/sales.csv

pipelines:
  filtered_sales:
    source: sales_data
    table: sales
    filters:
      - type: widget
        field: region
      - type: widget
        field: product

layouts:
  - title: Sales Overview
    pipeline: filtered_sales
    views:
      - type: hvplot
        kind: line
        x: date
        y: revenue
      - type: table
        page_size: 20
```

Deploy with one command:

```bash
lumen serve dashboard.yaml --show
```

## YAML vs Python

Lumen offers two approaches to building dashboards:

| Aspect | YAML | Python |
|--------|------|--------|
| **Syntax** | Simple configuration | Python code |
| **Learning curve** | Beginner-friendly | Requires Python knowledge |
| **Flexibility** | Good for common patterns | Full programmatic control |
| **Best for** | Standard dashboards | Complex custom applications |
| **Documentation** | This guide | [Python API Guide](python-api.md) |

Most users start with YAML and add Python only when needed.

## Getting help

- **Stuck?** Check the [Deployment guide](deployment.md) for validation
- **Need examples?** Each guide includes working code samples
- **Want to extend?** See the [Customization guide](customization.md)

## Next steps

**New to specs?** Start with the [Build Dashboard with Spec tutorial](../../tutorials/build_dashboard_with_spec.md) to create your first dashboard in 15 minutes.

**Returning users?** Jump to the guide that matches your current task using the navigation above.
