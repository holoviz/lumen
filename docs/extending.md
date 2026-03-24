# :material-puzzle-plus: Extending Lumen

Lumen's core building blocks—[Sources](configuration/sources.md), [Agents](configuration/agents.md), [Tools](configuration/tools.md), and [Analyses](configuration/analyses.md)—are all subclassable Python classes. If you've built a custom component and want to share it as a reusable, installable Python package, the **Panel Extension Copier Template** gives you a fully configured project scaffold to do just that.

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
