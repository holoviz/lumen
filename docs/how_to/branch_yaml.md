# Branch a pipeline in YAML

:::{admonition} What does this guide solve?
:class: important
This guide shows you how to build branching pipelines in YAML that allow you have some shared and some separate processing steps on the same source data.
:::

## Overview
 A Lumen Pipeline consists of Filters and Transforms, that is typically applied to a data Source to drive one or more visual outputs. However, a Pipeline can also be leveraged as a standalone component to encapsulate multiple data processing steps.

 For instance, you may want to display a table of filtered data alongside a view of aggregated data.
