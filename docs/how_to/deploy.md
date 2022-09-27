# Deploy a dashboard

:::{admonition} What does this guide solve?
:class: important
Whether you are still developing or have already completed your specification file, deployment creates a visual instantiation of your dashboard.
:::

Run the code below to start a Lumen server that displays your dashbboard in a browser window. Change `<dashboard.yaml>` to the path of your YAML specification file.


```console
lumen serve <dashboard.yaml> --show
```

## Deploy during development

While developing with Lumen, it is a great idea to visualize the progress of your dashboard. The code below uses `--autoreload` to refresh the view every time you save your YAML specification file.

```console
lumen serve <dashboard.yaml> --show --autoreload
```
