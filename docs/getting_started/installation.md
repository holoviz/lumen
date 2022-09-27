# {octicon}`desktop-download;2em;sd-mr-1` Installation

Lumen is a monitoring solution written in Python and built on [Panel](https://panel.holoviz.org/) and works on Linux, Windows, and Mac.
The recommended way to install `lumen` is using the [conda](http://conda.pydata.org/docs/) command provided by [Anaconda](https://www.anaconda.com) or [Miniconda](http://conda.pydata.org/miniconda.html), but installing `lumen` from [Python.org](https://www.python.org/) with [PyPi](https://pypi.org/) is also an option.

## Installing Python

:::{note}
This step can be skipped if you already have Python installed on your computer.
:::


1. Click the button below with your desired method for installing.
2. Download the installer matching your operating system.
3. Double-click on the installer and follow the instructions.

::::{grid} 3
:gutter: 3

:::{grid-item-card} Anaconda
:link: https://www.anaconda.com/products/distribution#Downloads
:text-align: center
:::

:::{grid-item-card} Miniconda
:link: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
:text-align: center
:::

:::{grid-item-card} Python.org
:link: https://www.python.org/downloads
:text-align: center
:::
::::

## Installing Lumen
1. Open up a terminal, command prompt, or Powershell based on your operating system.
2. Run the following command, which will install Lumen with all its dependencies.
3. When the command finishes, run `lumen --version` in the terminal and check that the version is {{version}}.
    - If this is not the case, you are not running the latest version, which may cause problems running the following examples.


::::{tab-set}
:::{tab-item} Conda
:sync: conda

``` bash
conda install -c pyviz -c conda-forge lumen -y
```

:::
:::{tab-item} Pip
:sync: pip

``` bash
pip install lumen
```
:::
::::


## Running _Getting Started_

To run the _Getting Started_ guide, some other dependencies are needed. These should be installed the same way as done in the previous section.

::::{tab-set}
:::{tab-item} Conda
:sync: conda

``` bash
conda install -c pyviz -c conda-forge intake hvplot -y
```

:::
:::{tab-item} Pip
:sync: pip

``` bash
pip install intake hvplot
```
:::
::::
