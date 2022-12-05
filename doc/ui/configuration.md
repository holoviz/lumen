# Configuration

Configuring the Lumen Builder UI is quite straightforward. In the directory where you will be launching the `lumen builder` from create a `components` directory with the following structure.

```
.
│   config.yml
│───components
│  │───dashboards
│  │      example.yml
│  │      example.png
│  │      ...
│  │───launchers
│  │      launcher.yml
│  │      ...
│  │───sources
│  │      source.yml
│  │      source.png
│  │      ...
│  │───variables
│  │      vars.yml
│  │      ...
│  │───views
│  │      view.yml
│  │      view.png
│  │      ...
```


:::{admonition} Note
:class: success

The first time you launch `lumen builder` in a directory it will set up this directory for you.
:::
