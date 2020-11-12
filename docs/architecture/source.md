# Sources

The [Source](lumen.sources.Source) classes are the data provider for
the entire dashboard. They handle querying some remote source such as
a REST server, a website or some other provider and provide methods to
return a set of tables (using the `.get` method) and JSON schemas
describing the tables (using the `.get_schema` method).

```{eval-rst}
.. autoclass:: lumen.sources.Source
   :members:
```

## Source types

```{eval-rst}
.. autoclass:: lumen.sources.JoinedSource
```

```{eval-rst}
.. autoclass:: lumen.sources.FileSource
```

```{eval-rst}
.. autoclass:: lumen.sources.PanelSessionSource
```

```{eval-rst}
.. autoclass:: lumen.sources.RESTSource
```

```{eval-rst}
.. autoclass:: lumen.sources.WebsiteSource
```
