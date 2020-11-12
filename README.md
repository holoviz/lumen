# Lumen

*Illuminate your data*

[![Build Status](https://travis-ci.com/holoviz/monitor.svg?branch=master)](https://travis-ci.com/holoviz/monitor)
## Purpose

The Lumen project provides a framework to build dashboards from a
simple yaml specification. It is designed to query information from
any source, filter it in various ways and then provide views of that
information, which can be anything from a simply indicator to a table
or plot.

A Lumen dashboard can be configured using a minimal yaml specification
configuring the data source, filters and views, making it easy to view
and monitor your data.

## Architecture

The library is organized into a small number of simply object types including:

* `Source`: A `Source` provides any number of tables along with a JSON schema describing the contents of those tables.
* `Filter`: A `Filter` object is given the schema of a field in one of the tables and generates queries which filter the data supplied by a `Source`. 
* `View`: A `View` can query a table from a `Source` and generates a viewable representation.
* `Transform`: A `Transform` can apply arbitrary transformation to the tables.

All of these base types can be easily subclassed to provide custom data sources, filters, transforms and views.
