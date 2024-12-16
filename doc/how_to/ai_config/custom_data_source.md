# Custom Data Sources

```{admonition} What does this guide solve?
---
class: important
---
This guide shows you how to configure custom data sources for Lumen AI.
```

## Overview

We will be using a local LLM to understand how to load custom data sources in Lumen. You do not need
to use a local LLM, and can instead opt for using one you have an API key for, ensure your API key
is in the environment of the terminal you run your commands in.

## Local and remote files using the command line

You can download the standard penguins data set
[here](https://datasets.holoviz.org/penguins/v1/penguins.csv). To start Lumen AI, run the following
command (replacing the path where you downloaded the data to).

```bash
lumen-ai serve penguins.csv --provider llama --show
```

If instead you do not want to download data, you can tell Lumen where the data is on the web, and
start a chat.

```bash
lumen-ai serve "https://datasets.holoviz.org/penguins/v1/penguins.csv" --provider llama --show
```

## Local and remote files using a Panel app

Download the [earthquakes](https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv) dataset and
make a note of where it is on your system. Create a file called `app.py` and update the path to
where the earthquakes data was downloaded to. We will use both a local file, and a remote file with
the app.

```python
# app.py
import lumen.ai as lmai
import panel as pn

pn.extension("vega")
llm = lmai.llm.Llama()

lmai.ExplorerUI(
    [
        "/LOCAL/PATH/TO/earthquakes.csv",
        "https://datasets.holoviz.org/penguins/v1/penguins.csv",
    ],
    llm=llm,
    agents=[lmai.agents.SQLAgent, lmai.agents.VegaLiteAgent],
).servable()
```

Run the `app.py` file with the following command.

```bash
panel serve app.py --show
```
