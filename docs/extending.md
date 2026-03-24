# :material-puzzle-plus: Extending Lumen

Lumen is designed to be extensible. While the built-in sources, agents, tools, and analyses cover a wide range of use cases, you may want to create custom components that integrate deeply with your own data infrastructure or domain logic.

---

## Custom Components

Lumen's core building blocks—[Sources](configuration/sources.md), [Agents](configuration/agents.md), [Tools](configuration/tools.md), and [Analyses](configuration/analyses.md)—are all subclassable Python classes. You can override or extend any of them and pass your custom class directly to `ExplorerUI`.

For example, a minimal custom source:

```python
from lumen.sources import Source
import pandas as pd

class MyCustomSource(Source):
    """A source that loads data from a proprietary API."""

    source_type = 'my_custom'

    def get(self, table, **query):
        # Replace with your own data fetching logic
        return pd.DataFrame({'col': [1, 2, 3]})

    def get_schema(self, table=None):
        return {'my_table': {'col': {'type': 'integer'}}}
```

Then use it like any built-in source:

```python
import lumen.ai as lmai

source = MyCustomSource()
ui = lmai.ExplorerUI(data=source)
ui.servable()
```

---

## Building a Distributable Extension

If your custom component is mature enough to share—with colleagues, as an open-source package, or on PyPI—consider packaging it as a proper Python extension using the **Panel Extension Copier Template**.

### What is the Panel Extension Copier Template?

The [`copier-template-panel-extension`](https://github.com/panel-extensions/copier-template-panel-extension) is a [Copier](https://copier.readthedocs.io/en/stable/) template that scaffolds a fully configured Python package for Panel and Lumen extensions. It gives you:

- **pytest** for testing
- **MkDocs + mkdocstrings** for automatic API docs hosted on GitHub Pages
- **GitHub Actions** CI/CD for automated testing and publishing
- **Pixi** for reproducible environment management

### Quickstart

Make sure you have [Pixi](https://pixi.sh) installed, then run:

```bash
pixi exec --spec copier --spec ruamel.yaml -- \
  copier copy --trust \
  https://github.com/panel-extensions/copier-template-panel-extension \
  lumen-name-of-extension
```

You'll be prompted for a few details such as extension type, project slug, and author info. Choose **Lumen** as the extension type when asked, and select `py311` or higher for the minimum Python version—Lumen requires Python 3.11+.

From there, follow the template's [step-by-step guide](https://github.com/panel-extensions/copier-template-panel-extension#getting-started) to set up GitHub Pages docs, link to PyPI, and publish your first release with a git tag.

---

## Further Reading

- [Data Sources](configuration/sources.md) — built-in sources and how to configure them
- [Agents](configuration/agents.md) — customizing agent behavior
- [Tools](configuration/tools.md) — adding domain-specific tools
- [Analyses](configuration/analyses.md) — deterministic analysis functions
- [Panel Extension Copier Template on GitHub](https://github.com/panel-extensions/copier-template-panel-extension)
