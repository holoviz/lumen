# Welcome to Lumen!

The Lumen project consists of two main pieces:

- **Lumen Core**: A framework for visual analytics, making it possible to build complex data processing pipelines, plots and entire dashboards using a declarative specification.
- **Lumen AI**: An extensible Agent based "chat-with-data" framework.

Together these pieces make it possible to perform complex data analysis using natural language, and then export the results, continue the analysis in a notebook or assemble the results into a dashboard using a drag-and-drop interface.

### Lumen AI

<img src="./_static/ai-diagram.png" alt="Lumen AI Diagram" width="100%"/>

Lumen AI provides a framework for chatting with data. It interfaces with the Lumen data sources providing the ability to connect with your database or data lake and unlock insights without writing code.

- Generate complex SQL queries to analyze the data
- Generate charts & powerful data tables or entire dashboards.
- Automatically summarize the key results and insights.
- Define custom analyses to generate deep insights tailored to your domain.

### Lumen Core

<img src="./_static/diagram.png" width="100%">

The power of Lumen comes from the ability to leverage the powerful data intake, data processing and data visualization libraries available in the PyData ecosystem.

- **Data Intake**: A flexible system for declaring data sources with strong integration with SQL, DuckDB and familiar Python DataFrame libraries. This allows Lumen to query data from a wide range of sources including many file formats such as CSV or Parquet but also SQL and many others and apply transformations where the data lives.
- **Data Proccessing**: Internally Lumen allows manipulating data in SQL or in Python as DataFrame objects. This allows Lumen to perform data transformations where the data lives (using SQL), while also providing the flexibility of familiar APIs for filtering and transforming data using [Pandas](https://pandas.pydata.org/) or scaling these transformations out to a cluster thanks to [Dask](https://dask.org/).
- **Data Visualization**: Since Lumen is built on [Panel](https://panel.holoviz.org) all the most popular plotting libraries and many other components such as powerful datagrids and BI indicators are supported.

The core strengths of Lumen include:

- **Flexibility**: The design of Lumen allows flexibly combining data intake, data processing and data visualization into a simple declarative pipeline.
- **Extensibility**: Every part of Lumen is designed to be extended letting you define custom Source, Filter, Transform and View components.
- **Scalability**: Lumen is designed with performance in mind and supports scalable Dask DataFrames out of the box, letting you scale to datasets larger than memory or even scale out to a cluster.
- **Security**: Lumen ships with a wide range of OAuth providers out of the box, making it a breeze to add authentication to your applications.

::::{grid} 1 2 2 4
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`desktop-download;2em;sd-mr-1` Installation
:link: getting_started/installation
:link-type: doc

Install Lumen in a few easy steps
:::

:::{grid-item-card} {octicon}`zap;2em;sd-mr-1` Exploration with Lumen AI
:link: getting_started/lumen_ai
:link-type: doc

How to explore your data with Lumen AI.
:::

:::{grid-item-card} {octicon}`tools;2em;sd-mr-1` Build a dashboard
:link: getting_started/build_dashboard
:link-type: doc

How to build a Lumen dashboard
:::

:::{grid-item-card} {octicon}`mortar-board;2em;sd-mr-1` Core concepts
:link: getting_started/core_concepts
:link-type: doc

Get an overview of the core concepts of Lumen
:::

:::{grid-item-card} {octicon}`git-commit;2em;sd-mr-1` Data Pipelines
:link: getting_started/pipelines
:link-type: doc

Discover how to build powerful data pipelines with with Lumen.
:::

::::

```{toctree}
---
hidden: true
---
Home <self>
Getting Started <getting_started/index>
How to <how_to/index>
Gallery <gallery/index>
Reference <reference/index>
Background <background/index>
```
