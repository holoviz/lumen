# REST Endpoint Spec

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
 
