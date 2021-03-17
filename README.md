# Lumen

*Illuminate your data*

<img src="https://raw.githubusercontent.com/holoviz/lumen/master/docs/_static/diagram.svg" width="100%">

|    |    |
| --- | --- |
| Build Status | [![Linux/MacOS/Windows Build Status](https://github.com/holoviz/lumen/workflows/pytest/badge.svg)](https://github.com/holoviz/lumen/actions/workflows/test.yml)
| Coverage | [![codecov](https://codecov.io/gh/holoviz/lumen/branch/master/graph/badge.svg)](https://codecov.io/gh/holoviz/lumen) |
| Latest dev release | [![Github tag](https://img.shields.io/github/v/tag/holoviz/lumen.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/lumen/tags) [![dev-site](https://img.shields.io/website-up-down-green-red/https/pyviz-dev.github.io/lumen.svg?label=dev%20website)](https://pyviz-dev.github.io/lumen/) |
| Latest release | [![Github release](https://img.shields.io/github/release/holoviz/lumen.svg?label=tag&colorB=11ccbb)](https://github.com/holoviz/lumen/releases) [![PyPI version](https://img.shields.io/pypi/v/lumen.svg?colorB=cc77dd)](https://pypi.python.org/pypi/lumen) [![lumen version](https://img.shields.io/conda/v/pyviz/lumen.svg?colorB=4488ff&style=flat)](https://anaconda.org/pyviz/lumen) [![conda-forge version](https://img.shields.io/conda/v/conda-forge/lumen.svg?label=conda%7Cconda-forge&colorB=4488ff)](https://anaconda.org/conda-forge/lumen) [![defaults version](https://img.shields.io/conda/v/anaconda/lumen.svg?label=conda%7Cdefaults&style=flat&colorB=4488ff)](https://anaconda.org/anaconda/lumen) |
| Docs | [![gh-pages](https://img.shields.io/github/last-commit/holoviz/lumen/gh-pages.svg)](https://github.com/holoviz/lumen/tree/gh-pages) [![site](https://img.shields.io/website-up-down-green-red/https/lumen.holoviz.org.svg)](https://lumen.holoviz.org) |
| Support | [![Discourse](https://img.shields.io/discourse/status?server=https%3A%2F%2Fdiscourse.holoviz.org)](https://discourse.holoviz.org/) |

## Purpose

The Lumen project provides a framework to build data-driven dashboards from a simple yaml specification. It is designed to query data from any source, filter it in various ways and then provide views of that information, which can be anything from a simply indicator to a table or a plot.

Since Lumen is built on [Panel](https://panel.holoviz.org) it supports a wide range of plotting libraries and other components to explore and visualize data. Thanks to integration with [Intake](https://intake.readthedocs.io/en/latest/), lightweight package for finding, investigating, loading and disseminating data, Lumen can query data from a wide range of sources including many file formats such as CSV or Parquet but also SQL and many others.

The library is organized into a small number of simply object types including:

* `Source`: A `Source` provides any number of tables along with a JSON schema describing the contents of those tables.
* `Filter`: A `Filter` object is given the schema of a field in one of the tables and generates queries which filter the data supplied by a `Source`. 
* `View`: A `View` can query a table from a `Source` and generates a viewable representation.
* `Transform`: A `Transform` can apply arbitrary transformation to the tables.

All of these base types can be easily subclassed to provide custom data sources, filters, transforms and views.

## Getting started

Lumen works with Python 3 and above on Linux, Windows, or Mac. The recommended way to install Lumen is using the [`conda`](https://conda.pydata.org/docs/) command provided by [Anaconda](https://docs.continuum.io/anaconda/install) or [`Miniconda`](https://conda.pydata.org/miniconda.html):

    conda install -c pyviz lumen

or using PyPI:

    pip install lumen

Once installed you will be able to start a Lumen server by running:

    lumen serve dashboard.yaml --show

This will open a browser serving the application or dashboard declared by your yaml file in a browser window. During development it is very helpful to use the `--autoreload` flag, which will automatically refresh and update the application in your browser window, whenever you make an edit to the dashboard yaml specification. In this way you can quickly iterate on your dashboard.

Try it out! Click on one of the examples below, copy the yaml specification and launch your first Lumen application.

## Examples

   <table>
     <tr>
       <td><a href="https://lumen.holoviz.org/gallery/bikes.html"><b>London Bike Points</b><br><img src="https://raw.githubusercontent.com/holoviz/lumen/master/docs/_static/bikes.png" /></a></td>
       <td><a href="https://lumen.holoviz.org/gallery/nyc_taxi.html"><b>NYC Taxi</b><br><img src="https://raw.githubusercontent.com/holoviz/lumen/master/docs/_static/nyc_taxi.png" /></a></td>
     </tr>
     <tr>
	   <td><a href="https://lumen.holoviz.org/gallery/penguins.html"><b>Palmer Penguins</b><br><img src="https://raw.githubusercontent.com/holoviz/lumen/master/docs/_static/penguins.png" /></a></td>
       <td><a href="https://lumen.holoviz.org/gallery/precip.html"><b>Precipitation</b><br><img src="https://raw.githubusercontent.com/holoviz/lumen/master/docs/_static/precip.png" /></a></td>
     <tr>
   </table>
