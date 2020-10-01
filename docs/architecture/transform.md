# Transforms

A [Transform](monitor.transforms.Transform) provides the ability to
transform the metric data supplied by a
[QueryAdaptor](monitor.adaptors.QueryAdaptor). Given a pandas
`DataFrame` it applies some transformation and returns another
`DataFrame`.

```{eval-rst}
.. autoclass:: monitor.transforms.Transform
   :members:
```

## Transform types

```{eval-rst}
.. autoclass:: monitor.transforms.HistoryTransform
```
