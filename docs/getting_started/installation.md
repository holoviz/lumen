# {octicon}`desktop-download;2em;sd-mr-1` Installation

Lumen works with Python 3 on Linux, Windows, and Mac.
The recommended way to install Lumen is using the [conda](http://conda.pydata.org/docs/) command provided by [Anaconda](https://www.anaconda.com) or [Miniconda](http://conda.pydata.org/miniconda.html). You can also install Lumen using [Pip](https://pypi.org/).

## Installing Lumen
1. Open up a terminal (Powershell if you are on Windows).
2. Run the following command, which will install Lumen with all its dependencies.
3. When the command finishes, run `lumen --version` in the terminal and check that the version is {{version}}.
    - If this is not the case, you are not running the latest version, which may cause problems. TODO: *guidance on what to do if version is incorrect*

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


## Optional dependencies

Lumen is very flexible and allows you to use components from various packages. Depending on what type of dashboard components you use, you may need to install additional packages. Error messages will help you determine if you are missing a package. For instance, if you see the following...
``` bash
Source component specification declared unknown type 'intake'.
```

...install the missing package in the same way you did lumen:

::::{tab-set}
:::{tab-item} Conda
:sync: conda

``` bash
conda install -c pyviz -c conda-forge intake -y
```

:::
:::{tab-item} Pip
:sync: pip

``` bash
pip install intake
```
:::
::::
