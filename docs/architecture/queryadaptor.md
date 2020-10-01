# QueryAdaptors

The [QueryAdaptor](lumen.adaptors.QueryAdaptor) classes are the data
provider for the entire dashboard. They handle querying some remote
source such as a REST server, a website or some other provider and
return the data in a common format for consumption by the
[MetricView](lumen.views.MetricView) classes.

```{eval-rst}
.. autoclass:: lumen.adaptors.QueryAdaptor
   :members:
```

## QueryAdaptor types

```{eval-rst}
.. autoclass:: lumen.adaptors.LiveWebsite
```

```{eval-rst}
.. autoclass:: lumen.adaptors.RESTAdaptor
```
