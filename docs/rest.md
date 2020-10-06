# REST Endpoint Specification

The REST specification defines the default format for data to be consumed by the dashboard. The API consists of three main endpoints providing access to the schema

- `schema`: Publishes a schema for each variable and all its associated indexes, the schema should follow the [JSON schema](https://json-schema.org/) specification:

    - Query: A query may define the variable to return the schema for:
        `{'variable': <variable>}`
    - Output:
    ```
    {
        <variable>: {
            <variable>: {'description': string, 'schema': object, 'label': string},
            <index>: {'description': string, 'schema': object, 'label': string},
            ...
        },
        ...
    }
    ```

- `data`: This endpoints returns the actual data, it allows querying by one or more indexes:

    - Query: A query must contain the variable to be returned and any number of index queries:
        `{'variable': <variable>, <index>: <value>, ...}`
    - Output: It will always return a list of records containing all the metric and filter values:
    ```
    [
        {<variable>: <value>, <index>: <value>, ...},
        {<variable>: <value>, <index>: <value>, ...},
        ...
    ]
    ```

- `dump`: Returns a complete dump of all data:

    - Query: None
    - Output:
    ```
    {
        <variable>: [
            {<variable>: <value>, <index>: <value>, ...},
            {<variable>: <value>, <index>: <value>, ...},
            ...
        ],
        ...
    }
    ```
