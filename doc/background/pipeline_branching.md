# Pipeline Branching

:::{admonition} What is this page about?
:class: important
This page discusses what a pipeline branch is and when you might need one.
:::

A Lumen [Pipeline]`lumen.pipeline.Pipeline` consists of `Filter` and `Transform` components. These manipulations of the data are typically applied to a data `Source` to drive one or more visual outputs. However, a [Pipeline]`lumen.pipeline.Pipeline` can also branch off of another [Pipeline]`lumen.pipeline.Pipeline` allowing for further data manipulation that exists only on the branch, while retaining the shared computations up to that branching point. This allows for the creation of `View` components of the same underlying source data, but at different steps in data processing. For instance, you may want to display a table of filtered data alongside a view of aggregated data.

## Related resources:
* [How to branch a Lumen pipeline](../how_to/data_processing/branch_pipeline)
