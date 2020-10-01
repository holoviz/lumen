# Architecture

```{toctree}
Architecture <self>
QueryAdaptor <queryadaptor>
MetricView <metric>
Filter <filter>
Transform <transform>
```

The Lumen dashboard can query information from any source using a so called `QueryAdaptor`. The `QueryAdaptor` can return any number of metrics and filters:

* `metric`: A `metric` is some quantitity that can be visualized
* `filter`: A `filter` is a variable that can be filtered by usually using a widget or by specifying a constant in the dashboard specification.

In addition to the actual values the `QueryAdaptor` should provide a JSON schema which describes the types of the `metric` and `filter` variables. 

The main `QueryAdaptor` types we envision to ship are:

- REST API: A well defined specification to publish metrics and filters
- File: A baseclass that can load data from a file (needs to be adapted for different file types)
  - CSV
  - Parquet
  - ...
- HTTP Status: A simple data source that returns the HTTP status of a web server
- Intake: A simple adaptor that can load data from an Intake data catalogue
- ...

Additionally we will want a plugin system (like in Intake) that allows providing additional QueryAdaptors.
