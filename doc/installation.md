# Installation

## Setup

Lumen works with Python 3 on Linux, Windows, and Mac.

The recommended way to install Lumen is using the [conda](https://docs.conda.io/projects/conda/en/latest/index.html) command that is included in the installation of [Anaconda or Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). To help you choose between Anaconda and Miniconda, review [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda). Completing the installation for either Anaconda or Miniconda will also install Python.

If you are not installing Anaconda or Miniconda, you can download Python directly from [Python.org](https://www.python.org/downloads/). In this case, you can install Lumen using [pip](https://pip.pypa.io/en/stable/), which comes with Python.

## Installing Lumen

1. Open up a terminal (Powershell if you are on Windows).
2. Run the following command, which will install Lumen with all its dependencies.
3. When the command finishes, run `lumen --version` in the terminal and check that the version is {{version}}.
    - If this is not the case, you are not running the latest version, which may cause problems.

::::{tab-set}

:::{tab-item} conda
:sync: conda

``` bash
conda install -c pyviz -c conda-forge lumen -y
```
:::

:::{tab-item} pip
:sync: pip

``` bash
pip install lumen
```
:::

::::

To use Lumen AI, you need to install additional dependencies in addition to the core dependencies. You can do this by running:

```bash
pip install 'lumen[ai]'
```

If you want to use a specific LLM provider, you can specify the provider name as a hyphenated suffix:

```bash
pip install 'lumen[ai-openai]'
pip install 'lumen[ai-mistralai]'
pip install 'lumen[ai-anthropic]'
```

For models that run locally on your machine and offline (after downloading and caching):

```bash
pip install 'lumen[ai-llama]'
```
