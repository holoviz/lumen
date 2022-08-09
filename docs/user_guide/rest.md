# REST Specification

The REST specification defines the default format for data to be consumed by the dashboard. The API consists of three main endpoints providing access to the schema

- `schema`: Publishes a schema for each variable and all its associated indexes, the schema should follow the [JSON schema](https://json-schema.org/) specification:

    - Query: A query may define the variable to return the schema for:
        `{'table': <table>}`
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

- `data`: This endpoints returns the actual data, it also allows filtering the data along the columns with query:

    - Query: A query must contain the table to be returned and may optionally provide a subset of columns and filters for the table columns:
        `{'table': <table>, 'columns': [<column>, ...], <column>: <value>, ...}`
    - Output: It will always return a list of records containing all the metric and filter values:
    ```
    [
        {<column>: <value>, <column>: <value>, ...},
        {<column>: <value>, <column>: <value>, ...},
        ...
    ]
    ```

- `dump`: Returns a complete dump of all data:

    - Query: None
    - Output:
    ```
    {
        <table>: [
            {<column>: <value>, <column>: <value>, ...},
            {<column>: <value>, <column>: <value>, ...},
            ...
        ],
        ...
    }
    ```
