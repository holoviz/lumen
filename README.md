# Lumen

*Illuminate your data*

|    |    |
| --- | --- |
| Build Status | [![Linux/MacOS/Windows Build Status](https://github.com/holoviz/lumen/actions/workflows/test.yaml/badge.svg)](https://github.com/holoviz/lumen/actions/workflows/test.yaml)
| Coverage | [![codecov](https://codecov.io/gh/holoviz/lumen/branch/main/graph/badge.svg)](https://codecov.io/gh/holoviz/lumen) |
| Latest dev release | [![Github tag](https://img.shields.io/github/v/tag/holoviz/lumen.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/lumen/tags) [![dev-site](https://img.shields.io/website-up-down-green-red/https/holoviz-dev.github.io/lumen.svg?label=dev%20website)](https://holoviz-dev.github.io/lumen/) |
| Latest release | [![Github release](https://img.shields.io/github/release/holoviz/lumen.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/lumen/releases) [![PyPI version](https://img.shields.io/pypi/v/lumen.svg?colorB=cc77dd)](https://pypi.python.org/pypi/lumen) [![lumen version](https://img.shields.io/conda/v/pyviz/lumen.svg?colorB=4488ff&style=flat)](https://anaconda.org/pyviz/lumen) [![conda-forge version](https://img.shields.io/conda/v/conda-forge/lumen.svg?label=conda%7Cconda-forge&colorB=4488ff)](https://anaconda.org/conda-forge/lumen) [![defaults version](https://img.shields.io/conda/v/anaconda/lumen.svg?label=conda%7Cdefaults&style=flat&colorB=4488ff)](https://anaconda.org/anaconda/lumen) |
| Docs | [![gh-pages](https://img.shields.io/github/last-commit/holoviz/lumen/gh-pages.svg)](https://github.com/holoviz/lumen/tree/gh-pages) [![site](https://img.shields.io/website-up-down-green-red/https/lumen.holoviz.org.svg)](https://lumen.holoviz.org) |
| Support | [![Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fdiscourse.holoviz.org)](https://discourse.holoviz.org/) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/ai-diagram-dark.png">
  <img src="https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/ai-diagram.png" alt="Lumen AI Diagram" width="100%"/>
</picture>

Lumen is a fully open-source and extensible agent based framework for chatting with data and for retrieval augmented generation (RAG). The declarative nature of Lumen's data model make it possible for LLMs to easily generate entire data transformation pipelines, visualizations and other many other types of output. Once generated the data pipelines and visual output can be easily serialized, making it possible to share them, to continue the analysis in a notebook and/or build entire dashboards.

- **Generate SQL**: Generate data pipelines on top of local or remote files, SQL databases or your data lake.
- **Provide context and embeddings**: Give Lumen access to your documents to give the LLM the context it needs.
- **Visualize your data**: Generate everything from charts to powerful data tables or entire **dashboards** using natural language.
- **Inspect, validate and edit results**: All LLM outputs can easily be inspected for mistakes, refined, and manually edited if needed.
- **Summarize results and key insights**: Have the LLM summarize key results and extract important insights.
- **Custom analyses, agents and tools**: Extend Lumen custom agents, tools, and analyses to generate deep insights tailored to your domain.

Lumen sets itself apart from other agent based frameworks in that it focuses on being fully open and extensible. With powerful internal primitives for expressing complex data transformations the LLM can gain insights into your datasets out-of-the box and can be further tailored with custom agents, analyses and tools to empower even non-programmers to perform complex analyses without having to code. The customization makes it possible to generate any type of output, allow the user and the LLM to perform analyses tailored to your domain and look up additional information and context easily. Since Lumen is built on [Panel](https://panel.holoviz.org) it can render almost any type of output with little to no effort, ensuring that even the most esoteric usecase is easily possible.

The declarative Lumen data model further sets it apart from other tools, making it easy for LLMs to populate custom components and making it easy for the user to share the results. Entire multi-step data transformation pipelines be they in SQL or Python can easily be captured and used to drive custom visualizations, interactive tables and more. Once generated the declarative nature of the Lumen specification allows them to be shared, reproducing them in a notebook or composing them through a drag-and-drop interface into a dashboard.

## Getting started

Lumen works with Python 3 and above on Linux, Windows, or Mac. The recommended way to install Lumen is using the [`conda`](https://conda.pydata.org/docs/) command provided by [Anaconda](https://docs.continuum.io/anaconda/install) or [`Miniconda`](https://conda.pydata.org/miniconda.html):

    conda install -c pyviz lumen

or using PyPI:

    pip install 'lumen[ai]'

Once installed you will be able to start a Lumen Explorer server by running (replace `data.csv` with your data):

    lumen-ai serve data.csv

Check out the [docs](https://lumen.holoviz.org/lumen_ai/getting_started/using_lumen_ai.html) for more details!
