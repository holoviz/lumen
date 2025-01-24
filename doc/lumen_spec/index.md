# Lumen Spec

Lumen Spec is a declarative data model designed to express data transformation pipelines in a simple, shareable format. It enables users to:

- Define multi-step data transformations in SQL or Python
- Create interactive visualizations and tables
- Share and reproduce analyses across notebooks
- Build dashboards through a drag-and-drop interface

The specification's declarative nature makes it easy to programmatically generate, modify and compose data pipelines while maintaining reproducibility.

---

## Getting Started

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;2em;sd-mr-1` Build a dashboard
:link: ./getting_started/build_dashboard
:link-type: doc

How to build a Lumen dashboard
:::

:::{grid-item-card} {octicon}`mortar-board;2em;sd-mr-1` Core concepts
:link: ./getting_started/core_concepts
:link-type: doc

Get an overview of the core concepts of Lumen
:::

:::{grid-item-card} {octicon}`git-commit;2em;sd-mr-1` Pipelines
:link: ./getting_started/pipelines
:link-type: doc

Get an overview of the core concepts of Lumen
:::

::::

## How-To

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;2em;sd-mr-1` Data Intake
:link: ./how_to/data_intake/index
:link-type: doc

How to configure data sources in Lumen dashboards
:::

:::{grid-item-card} {octicon}`git-commit;2em;sd-mr-1` Data Processing
:link: ./how_to/data_processing/index
:link-type: doc

How to specify data transformations in a Lumen dashboard
:::

:::{grid-item-card} {octicon}`graph;2em;sd-mr-1` Data Visualization
:link: ./how_to/data_visualization/index
:link-type: doc

How to configure visualization and other visual components in a Lumen dashboard
:::

:::{grid-item-card} {octicon}`codescan-checkmark;2em;sd-mr-1` Validate Specifications
:link: ./how_to/validation/index
:link-type: doc

How to validate the specification of a Lumen dashboard
:::

::::

```{toctree}
:titlesonly:
:hidden:
:maxdepth: 2

getting_started/index
how_to/index
```
