# How to Guides

Lumen's How to Guides provide step by step recipes for solving essential problems and tasks. They are more advanced than the Getting Started material and assume some knowledge of how Lumen works.

## [Validation](validation/index)

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Validate Specification
:link: validation/validate
:link-type: doc

Learn how to validate the YAML file that specifies a Lumen dashboard.
:::

::::

## [Data Intake](data_intake/index)

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Access Files
:link: data_intake/files
:link-type: doc

Learn how to use a local or remote file as a source for your dashboard.
:::

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Cache Data
:link: data_intake/cache
:link-type: doc

Learn how to locally cache data to speed up reloading from a remote Source.
:::

::::

## [Data Processing](data_processing/index)

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Build Pipelines with Python
:link: data_processing/pipeline_python
:link-type: doc

Learn the basics of how to build a Lumen Pipeline programmatically with Python.
:::

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Branch a Pipeline
:link: data_processing/branch_pipeline
:link-type: doc

Learn how to build branching pipelines, allowing for views of the same source data at different steps in processing.
:::

::::

## [Visualize and Deploy](data_visualization/index)

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Define Views
:link: data_visualization/views
:link-type: doc

Learn how to define views on your dashboard.
:::

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Build Dashboards with Python
:link: data_visualization/dashboard_python
:link-type: doc

Learn how to build a Lumen Dashboard programmatically with Python.
:::

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Deploy
:link: data_visualization/deploy
:link-type: doc

Learn how to deploy a visual instantiation of your dashboard.
:::

::::

## [Data Output](data_output/index)

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Enable Data Download
:link: data_output/download_data
:link-type: doc

Learn how to enable your dashboard's viewer to download data.
:::

::::

## [Advanced Topics](advanced/index)

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Reference Variables
:link: advanced/variables_and_references
:link-type: doc

How to use variables and references to link objects across the YAML file.
:::

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Access custom components
:link: advanced/local_components
:link-type: doc

Learn how to access local files that build custom components.
:::

:::{grid-item-card} {octicon}`workflow;2.5em;sd-mr-1` Configure Authentication
:link: advanced/auth
:link-type: doc

Learn how to configure authentication for your dashboard.
:::

::::

```{toctree}
:titlesonly:
:hidden:
:maxdepth: 2

Validation<validation/index>
Data intake<data_intake/index>
Data processing<data_processing/index>
Visualize and Deploy<data_visualization/index>
Data output<data_output/index>
Advanced topics<advanced/index>
```
