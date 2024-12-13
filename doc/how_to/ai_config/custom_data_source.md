# Custom Data Sources

```{admonition} What does this guide solve?
---
class: important
---
This guide shows you how to configure custom data sources for Lumen AI.
```

## Overview

We will be using a local LLM to understand how to load custom data sources in Lumen. You do not need
to use a local LLM, and can instead opt for using one you have an API key for. Lumen currently
supports the following LLM providers.

- OpenAI
- Anthropic
- MistralAI
- AzureOpenAI
- AzureMistralAI
- Llama

Ensure your API key is in the environment of the terminal you run your commands in.

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
