# Architecture

```{toctree}
Architecture <self>
Source <source>
View <view>
Filter <filter>
Transform <transform>
```

The library is organized into a small number of simply object types including:

* `Source`: A `Source` object provides any number of variables along
  with associated indexes from some data source. Additionally it also
  returns a schema describing the data.
* `Filter`: A `Filter` object filters the data along any of the
  indexes.
* `View`: A `View` takes the queried data for a variable and
  visualizes it as a Panel object.
* `Transform`: A `Transform` can apply arbitrary transformation on the
  queried data.

The information that feeds the filters and views are queried from a `Source` object. The `Source` should return any number of variables with associated indexes:

* `variable`: A `variable` is some quantitity that is being visualized.
* `index`: An `index` is a variable that can be filtered on usually using a widget or by specifying a constant in the dashboard specification.

The main `Source` types provided by Lumen include:

- REST API: A well defined specification to publish metrics and filters
- File: A baseclass that can load data from a file:
  - CSV
  - Parquet
- HTTP Status: A simple data source that returns the HTTP status of a web server
- Intake: A simple adaptor that can load data from an Intake data catalogue

All of these base types can be subclassed to provide custom implementations.
