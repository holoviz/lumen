# Deploy a dashboard

```{admonition} What does this guide solve?
---
class: important
---
Whether you are still developing or have already completed your specification file, deployment creates a visual instantiation of your dashboard.
```

Run the code below to start a Lumen server that displays your dashboard in a browser window. Change `<dashboard.yaml>` to the path of your YAML specification file.


``` bash
lumen serve <dashboard.yaml> --show
```

If you are developing a dashboard using a Python script (see [Lumen in Python guide](pipeline_python)) instead of a YAML file, you can use the same deployment approach by supplying the path to the Python script.

```bash
lumen serve <dashboard.py> --show
```

## Deploy during development

While developing with Lumen, it is a great idea to visualize the progress of your dashboard. The code below uses `--autoreload` to refresh the view every time you save your YAML specification file.

``` bash
lumen serve <dashboard.yaml> --show --autoreload
```
