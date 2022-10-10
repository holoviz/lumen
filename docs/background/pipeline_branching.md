# Pipeline Branching

A Lumen `Pipeline` consists of `Filters` and `Transforms`. These manipulations of the data are typically applied to a data `Source` to drive one or more visual outputs. However, a `Pipeline` can also branch off of another `Pipeline` allowing for further data manipulation that exists only on the branch, while retaining the shared computations up to that branching point. This allows for the creation of `Views` of the same underlying source data, but at different steps in data processing. For instance, you may want to display a table of filtered data alongside a view of aggregated data.

## Related resources:
* [How to branch a Lumen pipeline in Python](../how_to/chain_python.md)
* [How to branch a Lumen pipeline in YAML](../how_to/branch_yaml.md)
