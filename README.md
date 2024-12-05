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

## Why Lumen?

The Lumen project consists of two main pieces:

- **Lumen Core**: A framework for visual analytics, making it possible to build complex data processing pipelines, plots and entire dashboards using a declarative specification.
- **Lumen AI**: An extensible Agent based "chat-with-data" framework.

Together these pieces make it possible to perform complex data analysis using natural language, and then export the results, continue the analysis in a notebook or assemble the results into a dashboard using a drag-and-drop interface.

### Lumen AI

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/holoviz/lumen/ai-docs/doc/_static/ai-diagram-dark.png">
  <img src="https://raw.githubusercontent.com/holoviz/lumen/ai-docs/doc/_static/ai-diagram.png" alt="Lumen AI Diagram" width="100%"/>
</picture>

Lumen AI provides a framework for chatting with data. It interfaces with the Lumen data sources providing the ability to connect with your database or data lake and unlock insights without writing code.

- Generate complex SQL queries to analyze the data
- Generate charts & powerful data tables or entire dashboards.
- Automatically summarize the key results and insights.
- Define custom analyses to generate deep insights tailored to your domain.

### Lumen Core

<img src="https://raw.githubusercontent.com/holoviz/lumen/main/doc/_static/diagram.png" width="100%">

The power of Lumen comes from the ability to leverage the powerful data intake, data processing and data visualization libraries available in the PyData ecosystem.

- **Data Intake**: A flexible system for declaring data sources with strong integration with SQL, DuckDB and familiar Python DataFrame libraries. This allows Lumen to query data from a wide range of sources including many file formats such as CSV or Parquet but also SQL and many others and apply transformations where the data lives.
- **Data Proccessing**: Internally Lumen allows manipulating data in SQL or in Python as DataFrame objects. This allows Lumen to perform data transformations where the data lives (using SQL), while also providing the flexibility of familiar APIs for filtering and transforming data using [Pandas](https://pandas.pydata.org/) or scaling these transformations out to a cluster thanks to [Dask](https://dask.org/).
- **Data Visualization**: Since Lumen is built on [Panel](https://panel.holoviz.org) all the most popular plotting libraries and many other components such as powerful datagrids and BI indicators are supported.

The core strengths of Lumen include:

- **Flexibility**: The design of Lumen allows flexibly combining data intake, data processing and data visualization into a simple declarative pipeline.
- **Extensibility**: Every part of Lumen is designed to be extended letting you define custom Source, Filter, Transform and View components.
- **Scalability**: Lumen is designed with performance in mind and supports scalable Dask DataFrames out of the box, letting you scale to datasets larger than memory or even scale out to a cluster.
- **Security**: Lumen ships with a wide range of OAuth providers out of the box, making it a breeze to add authentication to your applications.

## Getting started

Lumen works with Python 3 and above on Linux, Windows, or Mac. The recommended way to install Lumen is using the [`conda`](https://conda.pydata.org/docs/) command provided by [Anaconda](https://docs.continuum.io/anaconda/install) or [`Miniconda`](https://conda.pydata.org/miniconda.html):

    conda install -c pyviz lumen

or using PyPI:

    pip install lumen[ai]

Once installed you will be able to start a Lumen server by running:

    lumen serve dashboard.yaml --show

This will open a browser serving the application or dashboard declared by your yaml file in a browser window. During development it is very helpful to use the `--autoreload` flag, which will automatically refresh and update the application in your browser window, whenever you make an edit to the dashboard yaml specification. In this way you can quickly iterate on your dashboard.

Try it out! Click on one of the examples below, copy the yaml specification and launch your first Lumen application.
