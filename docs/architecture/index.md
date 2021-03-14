# Architecture

The library is organized into a small number of simple object types including:

* `Source`: A `Source` provides any number of tables along with a JSON schema describing the contents of those tables.
* `Filter`: A `Filter` object is given the schema of a field in one of the tables and generates queries which filter the data supplied by a `Source`. 
* `View`: A `View` can query a table from a `Source` and generates a viewable representation.
* `Transform`: A `Transform` can apply arbitrary transformation to the tables.

All of these base types can be easily subclassed to provide custom data sources, filters, transforms and views.

```{toctree}
Architecture <self>
Source <source>
View <view>
Filter <filter>
Transform <transform>
```
