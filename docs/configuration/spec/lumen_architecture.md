# Lumen's architecture

A Lumen specification uses YAML to structure data exploration and dashboard settings. Data exploration sections are required for the dashboard, while dashboard settings are optional.

## Data exploration sections

The core sections that drive data handling:

**sources**
: Defines where the dashboard finds its data

**pipelines**
: Manipulates data with filters and transforms

**layouts**
: Presents manipulated data with views

## Dashboard settings sections

Optional sections that configure the overall dashboard behavior:

**config**
: Applies settings to the entire dashboard (title, theme, layout)

**defaults**
: Overrides default parameters for filters, sources, transforms, and views

**variables**
: Creates global variables referenced throughout the YAML

**auth**
: Configures authentication for dashboard access
