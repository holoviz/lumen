# Welcome to Lumen!

<img src="./_static/diagram.svg" width="100%">

The Lumen project provides a framework to build data-driven dashboards
from a simple yaml specification. It is designed to query data from
any source, filter it in various ways and then provide views of that
information, which can be anything from a simple indicator to a table
or a plot.

Since Lumen is built on [Panel](https://panel.holoviz.org) it supports
a wide range of plotting libraries and other components to explore and
visualize data. Thanks to integration with
[Intake](https://intake.readthedocs.io/en/latest/), a lightweight
package for finding, investigating, loading and disseminating data,
Lumen can query data from a wide range of sources including many file
formats such as CSV or Parquet but also SQL and many others.

Lumen is organized into a small number of simple object types which
can be easily subclassed and extended:

* `Source`: A `Source` provides any number of tables along with a JSON schema describing the contents of those tables.
* `Filter`: A `Filter` object is given the schema of a field in one of the tables and generates queries which filter the data supplied by a `Source`.
* `View`: A `View` can query a table from a `Source` and generates a viewable representation.
* `Transform`: A `Transform` can apply arbitrary transformation to the tables.

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
       <td><a href="./gallery/bikes.html"><b>London Bike Points</b><br><img src="./_static/bikes.png" /></a></td>
       <td><a href="./gallery/nyc_taxi.html"><b>NYC Taxi</b><br><img src="./_static/nyc_taxi.png" /></a></td>
     </tr>
     <tr>
	   <td><a href="./gallery/penguins.html"><b>Palmer Penguins</b><br><img src="./_static/penguins.png" /></a></td>
       <td><a href="./gallery/precip.html"><b>Precipitation</b><br><img src="./_static/precip.png" /></a></td>
     <tr>
   </table>



```{toctree}
---
maxdepth: 2
caption: Contents
---
Home <self>
Dashboard Specification <dashboard>
Gallery <gallery/index>
Architecture <architecture/index>
REST Specification <rest>
```
