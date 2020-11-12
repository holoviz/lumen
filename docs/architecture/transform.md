# Transforms

A [Transform](lumen.transforms.Transform) provides the ability to
transform the tables supplied by a
[Source](lumen.sources.Source). Given a pandas `DataFrame` it applies
some transformation and returns another `DataFrame`.

```{eval-rst}
.. autoclass:: lumen.transforms.Transform
   :members:
```

## Transform types

```{eval-rst}
.. autoclass:: lumen.transforms.Aggregate
```

```{eval-rst}
.. autoclass:: lumen.transforms.Columns
```

```{eval-rst}
.. autoclass:: lumen.transforms.HistoryTransform
```

```{eval-rst}
.. autoclass:: lumen.transforms.Query
```

```{eval-rst}
.. autoclass:: lumen.transforms.Sort
```

```{eval-rst}
.. autoclass:: lumen.transforms.Stack
```

```{eval-rst}
.. autoclass:: lumen.transforms.Unstack
```
