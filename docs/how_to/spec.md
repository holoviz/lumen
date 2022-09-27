# Write a specification file

:::{admonition} What does this guide solve?
:class: important
This guide walks through the creation of a Lumen `YAML specification file`.
:::

## Overview
The Lumen YAML file is a simple and human readable interface to specify all the aspects about a dashboard. Very briefly, YAML uses consistent whitespace to denote structure, and most lines either create an association (`key: value`), or create a list with a leading hyphen (`-`). For more on YAML, we recommend this [quick YAML guide](https://learnxinyminutes.com/docs/yaml/).

The three primary sections of this specification file are:
1. `sources`: list your data sources
2. `pipeline`: specify how you want the data to be `filtered` and `transformed`
3. `targets`: create the views (e.g. table, plot) for your dashboard

These core sections should be unindented as the top of the hierarchy:

```{code-block} YAML
sources:
  ...:
    ...: ...
pipelines:
  ...:
    ...: ...
targets:
  - ...: ...
    ...: ...
```

There are some other top level section that you may or may not need. We will get to theses later on, but just to give you a taste:

5. `config`: apply settings for the whole dashboard
6. `defaults`: provide default parameters for other sections
7. `variable`: create variables to reference throughout the YAML
8. `auth`: add authentication to your dashboard

## Specifying data sources

