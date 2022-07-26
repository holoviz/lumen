# Architecture

The library is organized into a small number of simple object types including:

* `Source`: A `Source` provides any number of tables along with a JSON schema describing the contents of those tables.
* `Filter`: A `Filter` object is given the schema of a field in one of the tables and generates queries which filter the data supplied by a `Source`.
* `Transform`: A `Transform` can apply arbitrary transformation to the tables.
* `Pipeline`: A `Pipeline` encapsulates a `Source` and any number of `Filter` and `Transform` components into a single data processing pipeline that can be used to drive one or more `View` components.
* `View`: A `View` can query a table from a `Source` and generates a viewable representation.

All of these base types can be easily subclassed to provide custom data sources, filters, transforms and views.

```{toctree}
Architecture <self>
Source <source>
Filter <filter>
Transform <transform>
Pipeline <pipeline>
View <view>
```
