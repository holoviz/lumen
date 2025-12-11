# Deploy dashboards

Deploy your dashboard to make it accessible as a standalone application.

## Deploy from YAML

Serve your YAML specification file using:

```bash
lumen serve <dashboard.yaml> --show
```

Replace `<dashboard.yaml>` with the path to your specification file.

## Deploy from Python

If you built your dashboard using a Python script, serve it the same way:

```bash
lumen serve <dashboard.py> --show
```

## Develop with auto-reload

While developing, use `--autoreload` to refresh the dashboard whenever you save changes:

```bash
lumen serve <dashboard.yaml> --show --autoreload
```

This lets you see updates in real-time as you modify your specification.
