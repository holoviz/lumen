# Monitor


## Purpose

The purpose of the Monitor dashboard is to watch a number of metrics which are obtained from some source, this could be from a REST endpoint, from a file or a simple uptime status on an HTTP server. 

## Architecture

The Monitor dashboard can query information from any source using a so called `QueryAdaptor`. The `QueryAdaptor` can return any number of metrics and filters:

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

## REST Endpoint Spec

The REST specification that we will publish alongside the monitor will have a small number of well defined endpoints which provide access to the JSON schema. 

- `metrics`: Publishes a schema for each metric and all its associated filter variables, the schema should follow the [JSON schema](https://json-schema.org/) specification:

```
{
   <metric_name>: {
       <metric>: {'description': string, 'schema': object, 'label': string},
       <filter>: {'description': string, 'schema': object, 'label': string},
       ...
   },
   ...
}
```

- `metric`: This endpoints returns the actual data, it allows querying by one or more variables

    - Query: A query must contain the metric to be returned and any number of filter queries:
        `{'metric': <metric_name>, <filter>: <value>, ...}`
    - Output: It will always return a list of records containing all the metric and filter values:
    ```
    [
        {<metric_name>: <value>, <filter_name1>: <value>, ...},
        {<metric_name>: <value>, <filter_name1>: <value>, ...},
        ...
    ]
    ```
- `dump`: Returns a complete dump of all data:
    
    - Query: None
    - Output:
    ```
    {
        <metric_name>: [
            {<metric_name>: <value>, <filter_name1>: <value>, ...},
            {<metric_name>: <value>, <filter_name1>: <value>, ...},
            ...
        ], 
        ...
    }
    ```
    
    
## Dashboard spec

The Monitor dashboard will be configured by a `dashboard.yml` file. The specification for the dashboard will consist of the following:

- `targets`: This is the list of targets to monitor
    - `adaptor`: The `QueryAdaptor` to use to read the target
    - `filters`: Any filter specifications mapping from filter name to filter type
        - <filter_var>: 
            - `constant`: Declares any constant values to declare as a filter, e.g. when monitoring a specific facility
            - `widget`: Declares that a widget should be generated for this filter
            - `query`: Allows specifying the value as a URL query parameter to the dashboard
    - `indicators`: Specifies how to display the metrics, mapping from indicator type to metric type, certain indicators can have more than one metric or even filter as input
        - <indicator>: <metric>
  
