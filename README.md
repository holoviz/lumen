# Lumen

[![Build Status](https://travis-ci.com/holoviz/monitor.svg?branch=master)](https://travis-ci.com/holoviz/monitor)

## Purpose

The purpose of the Lumen project is to watch a number of metrics which are obtained from some source, this could be from a REST endpoint, from a file or a simple uptime status on an HTTP server.

## Architecture

The Lumen dashboard can query information from any source using a so called `Source`. The `Source` can return any number of metrics and filters:

* `metric`: A `metric` is some quantitity that can be visualized
* `filter`: A `filter` is a variable that can be filtered by usually using a widget or by specifying a constant in the dashboard specification.

In addition to the actual values the `Source` should provide a JSON schema which describes the types of the `metric` and `filter` variables. 

The main `Source` types we envision to ship are:

- REST API: A well defined specification to publish metrics and filters
- File: A baseclass that can load data from a file (needs to be adapted for different file types)
  - CSV
  - Parquet
  - ...
- HTTP Status: A simple data source that returns the HTTP status of a web server
- Intake: A simple adaptor that can load data from an Intake data catalogue
- ...

Additionally we will want a plugin system (like in Intake) that allows providing additional Sources.
