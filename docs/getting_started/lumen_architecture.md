# Lumen's Architecture

The specification file used by `lumen` is a [YAML](https://en.wikipedia.org/wiki/YAML) file. The specification can be divided into sections of data exploration and dashboard settings. As a rule of thumb, the data exploration sections are required for the dashboard, and dashboard settings are optional.

### Data Exploration

::::{grid} 2
:gutter: 3

:::{grid-item-card} `sources`
:link: ../user_guide/dashboard.html#sources
Where the app finds the data.

:::
:::{grid-item-card} `pipelines`
:link: ../user_guide/dashboard.html#pipelines
Manipulating the data with `filters` and `transforms`.
:::

::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} `layouts`
:link: ../user_guide/dashboard.html#layouts

The presentation of the manipulated data with `views`.
:::

::::


### Dashboard Settings

::::{grid} 2
:gutter: 3

:::{grid-item-card} `config`
:link: ../user_guide/dashboard.html#config

Settings that are applied to the whole dashboard.
:::

:::{grid-item-card} `defaults`
:link: ../user_guide/dashboard.html#defaults

Overriding default parameters.
:::

:::{grid-item-card} `variables`
:link: ../user_guide/dashboard.html#variables

Global variables that can be used throughout the YAML file.
:::

:::{grid-item-card} `auth`
:link: ../user_guide/dashboard.html#auth

Authentication for the dashboard.
:::
::::
