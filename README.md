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

* `Source`: A `Source` object can provide one or more tables along with a JSON schema describing the columns or fields in the tables.
* `Filter`: A `Filter` can filter the data along any field.
* `View`: A `View` queries a table from one of the sources and visualizes it in some form, e.g. as an indicator, plot or table.
* `Transform`: A `Transform` can apply arbitrary transformation on the tables before it is given to the `View`.

The tables that feed the filters, transforms and views are queried from a `Source` object. The `Source` may return any number of tables which can contain any number of fields and rows.

The main `Source` types provided by Lumen include:

- REST API: A well defined specification to publish metrics and filters
- File: A baseclass that can load data from a file:
  - CSV
  - Parquet
- HTTP Status: A simple data source that returns the HTTP status of a web server
- Intake: A simple adaptor that can load data from an Intake data catalogue
- Panel Session Info: Queries the Panel session_info REST endpoint and returns information about deployed dashboards.
