# Sources

The [Source](lumen.sources.Source) classes are the data provider for
the entire dashboard. They handle querying some remote source such as
a REST server, a website or some other provider and return the data in
a common format for consumption by the [View](lumen.views.View)
classes.

```{eval-rst}
.. autoclass:: lumen.sources.Source
   :members:
```

## Source types

```{eval-rst}
.. autoclass:: lumen.sources.RESTSource
```

```{eval-rst}
.. autoclass:: lumen.sources.WebsiteSource
```
