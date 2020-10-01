# Filters

The [Filter](monitor.filters.Filter) classes provide the ability to
query just a subset of the data provided by a
[QueryAdaptor](monitor.adaptors.QueryAdaptor). They therefore provide
a powerful mechanism to drill down into just a subset of the data.

The [Filter](monitor.filters.Filter) API is very simple:

```{eval-rst}
.. autoclass:: monitor.filters.Filter
   :members:
```

## Filter types


```{eval-rst}
.. autoclass:: monitor.filters.ConstantFilter
```

```{eval-rst}
.. autoclass:: monitor.filters.FacetFilter
```

```{eval-rst}
.. autoclass:: monitor.filters.WidgetFilter
```
