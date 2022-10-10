# Branch a pipeline in Python

:::{admonition} What does this guide solve?
:class: important
This guide will show you how to branch a pipeline in Python so that you can have different processing steps and views on the same source data.
:::

## Overview
See the Background page on [Pipeline Branching](../background/pipeline_branching.md).

As we will see below, the primary tool to create a branch of a Lumen pipeline in Python is to use the `pipeline.chain` method.

## Initiating the pipeline
Let's start by creating a pipeline up to a branching point. See the How to guide - [Build a dashboard in Python](ht_pipeline_python) - for a walk-through of these initial steps.

```python
from lumen.pipeline import Pipeline
from lumen.sources import FileSource

data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-28/penguins.csv'

pipeline = Pipeline(source=FileSource(tables={'penguins': data_url}), table='penguins')

pipeline.add_filter('widget', field='species')
pipeline.add_filter('widget', field='island')
pipeline.add_filter('widget', field='sex')
pipeline.add_filter('widget', field='year')

pipeline.data
```

## Branching the pipeline
At this point, we will create a branch of our pipeline by using the `pipeline.chain` method, and apply a new transform that aggregates the data, only on this branch. We assign the result to a new `Pipeline` variable so that we can refer to it separately from the original `pipeline`.

```python
from lumen.transforms import Aggregate

agg_pipeline = pipeline.chain(transforms=[Aggregate(method='mean', by=['species', 'year'])])

agg_pipeline.data
```
