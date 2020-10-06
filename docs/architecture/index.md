# Architecture

```{toctree}
Architecture <self>
Source <source>
View <view>
Filter <filter>
Transform <transform>
```

The Lumen dashboard is designed to query information from any source, filter it in various ways and then provide views of that information, which can be anything from a simply indicator to a table or plot.

The information that feeds the filters and views are queried from a `Source` object. The `Source` should return any number of variables with associated indexes:

* `variable`: A `variable` is some quantitity that is being visualized.
* `index`: An `index` is a variable that can be filtered on usually using a widget or by specifying a constant in the dashboard specification.

In addition to the actual values the `Source` should provide a JSON schema which describes the types of the `variable` and `index` values.

The main `Source` types we envision to ship are:

- REST API: A well defined specification to publish metrics and filters
- File: A baseclass that can load data from a file (needs to be adapted for different file types)
  - CSV
  - Parquet
  - ...
- HTTP Status: A simple data source that returns the HTTP status of a web server
- Intake: A simple adaptor that can load data from an Intake data catalogue
- ...

Additionally we will want a plugin system (like in Intake) that allows providing additional Sources.
