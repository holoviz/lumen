# How to Guides

Lumen's How to Guides provide step by step recipes for solving essential problems and tasks. They are more advanced than the Getting Started material and assume some knowledge of how Lumen works.

## Configuring Lumen AI

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cache;2.5em;sd-mr-1` Custom Data Sources
:link: ai_config/custom_data_source
:link-type: doc

Learn how to configure custom data sources, to connect with databases or local and remote files, when working with Lumen AI.
:::

:::{grid-item-card} {octicon}`tools;2.5em;sd-mr-1` Custom Agents
:link: ai_config/custom_agents
:link-type: doc

Learn how to create and configure custom agents to accomplish goals in Lumen AI.
:::

:::{grid-item-card} {octicon}`tasklist;2.5em;sd-mr-1` Custom Analyses
:link: ai_config/custom_analyses
:link-type: doc

Discover how to implement and output custom views within Lumen AI.
:::

:::{grid-item-card} {octicon}`plug;2.5em;sd-mr-1` Custom Tools
:link: ai_config/custom_tools
:link-type: doc

Explore how to develop and incorporate custom tools to provide additional context.
:::

:::{grid-item-card} {octicon}`comment-discussion;2.5em;sd-mr-1` Model Prompts
:link: ai_config/model_prompts
:link-type: doc

Understand how to define and manage model-specific prompts for replacing response models.
:::

:::{grid-item-card} {octicon}`note;2.5em;sd-mr-1` Template Prompts
:link: ai_config/template_prompts
:link-type: doc

Learn to create and use template prompts for influencing the AI's responses.
:::

::::

## Data Intake

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`link-external;2.5em;sd-mr-1` Access Files
:link: data_intake/files
:link-type: doc

Learn how to use a local or remote file as a source for your dashboard.
:::

:::{grid-item-card} {octicon}`database;2.5em;sd-mr-1` Cache Data
:link: data_intake/cache
:link-type: doc

Learn how to locally cache data to speed up reloading from a remote Source.
:::

::::

## Data Processing

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-commit;2.5em;sd-mr-1` Build Pipelines with Python
:link: data_processing/pipeline_python
:link-type: doc

Learn the basics of how to build a Lumen Pipeline programmatically with Python.
:::

:::{grid-item-card} {octicon}`git-branch;2.5em;sd-mr-1` Branch a Pipeline
:link: data_processing/branch_pipeline
:link-type: doc

Learn how to build branching pipelines, allowing for views of the same source data at different steps in processing.
:::

::::

## Visualize and Deploy

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`image;2.5em;sd-mr-1` Define Views
:link: data_visualization/views
:link-type: doc

Learn how to define views on your dashboard.
:::

:::{grid-item-card} {octicon}`graph;2.5em;sd-mr-1` Visualize Dashboards with Python
:link: data_visualization/dashboard_python
:link-type: doc

Learn how to build a Lumen Dashboard programmatically with Python.
:::

:::{grid-item-card} {octicon}`package-dependents;2.5em;sd-mr-1` Deploy
:link: data_visualization/deploy
:link-type: doc

Learn how to deploy a visual instantiation of your dashboard.
:::

::::

## Validation

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`bug;2.5em;sd-mr-1` Validate Specification
:link: validation/validate
:link-type: doc

Learn how to validate the YAML file that specifies a Lumen dashboard.
:::

::::

## Data Output

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;2.5em;sd-mr-1` Enable Data Download
:link: data_output/download_data
:link-type: doc

Learn how to enable your dashboard's viewer to download data.
:::

::::

## Advanced Topics

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`link;2.5em;sd-mr-1` Reference Variables
:link: advanced/variables_and_references
:link-type: doc

How to use variables and references to link objects across the YAML file.
:::

:::{grid-item-card} {octicon}`upload;2.5em;sd-mr-1` Access Custom Components
:link: advanced/local_components
:link-type: doc

Learn how to access local files that build custom components.
:::

:::{grid-item-card} {octicon}`unlock;2.5em;sd-mr-1` Configure Authentication
:link: advanced/auth
:link-type: doc

Learn how to configure authentication for your dashboard.
:::

:::{grid-item-card} {octicon}`zap;2.5em;sd-mr-1` Define Callbacks
:link: advanced/callbacks
:link-type: doc

Learn how to perform custom actions with callbacks.
:::

::::

```{toctree}
:titlesonly:
:hidden:
:maxdepth: 1

Configuring Lumen AI<ai_config/index>
Data Intake<data_intake/index>
Data Processing<data_processing/index>
Visualize and Deploy<data_visualization/index>
Validation<validation/index>
Data Output<data_output/index>
Advanced Topics<advanced/index>
```
