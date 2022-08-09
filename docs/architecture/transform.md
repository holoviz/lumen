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

## SQLTransform

A [SQLTransform](lumen.transforms.sql.SQLTransform) is a special type
of transform that operates on a [Source](lumen.sources.Source) that
supports SQL expressions. Instead of operating on the data it operates
on the SQL expression itself, thereby offloading potentially expensive
transformations onto the database.

```{eval-rst}
.. autoclass:: lumen.transforms.Transform
   :members:
```

### SQLTransform Types

```{eval-rst}
.. autoclass:: lumen.transforms.sql.SQLGroupBy
```

```{eval-rst}
.. autoclass:: lumen.transforms.sql.SQLLimit
```

```{eval-rst}
.. autoclass:: lumen.transforms.sql.SQLDistinct
```

```{eval-rst}
.. autoclass:: lumen.transforms.sql.SQLMinMax
```

```{eval-rst}
.. autoclass:: lumen.transforms.sql.SQLColumns
```

```{eval-rst}
.. autoclass:: lumen.transforms.sql.SQLFilter
```
