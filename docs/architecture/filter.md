# Filters

The [Filter](lumen.filters.Filter) classes provide the ability to
query just a subset of the data provided by a
[Source](lumen.sources.Source). They therefore provide a powerful
mechanism to drill down into just a subset of the data.

The [Filter](lumen.filters.Filter) API is very simple:

```{eval-rst}
.. autoclass:: lumen.filters.Filter
   :members:
```

## Filter types


```{eval-rst}
.. autoclass:: lumen.filters.ConstantFilter
```

```{eval-rst}
.. autoclass:: lumen.filters.FacetFilter
```

```{eval-rst}
.. autoclass:: lumen.filters.WidgetFilter
```
