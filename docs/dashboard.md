# Dashboard spec

The Monitor dashboard will be configured by a `dashboard.yml` file. The specification for the dashboard will consist of the following:

- `config`:
  - `title`: The title of the overall application
  - `template`: The template to use for the monitoring application
  - `ncols`: The number of columns to use in the grid of cards
- `endpoints`: This is the list of targets to monitor
  - `title`: The title of the monitoring endpoint
  - `adaptor`: The `QueryAdaptor` used to monitor an endpoint
    - `type`: The type of `QueryAdaptor` to use, e.g. 'rest' or 'live'
	- ...: Additional parameters for the `QueryAdaptor`
  - `metrics`: A list of metrics to monitor and display on the endpoint
    - `name`: The name of the metric
	- `type`: The type of `MetricView` to use for rendering the metric
	- ...: Additional parameters for the `MetricView`
  - `filters`: A list of `Filter` types to select a subset of the data
    - `name`: The name of the filter
	- `type`: The type of the `Filter` to use, e.g. 'constant', 'widget' or 'facet'
	- ...: Additional parameters for the `Filter`
  - `height`: The height of the card(s)
  - `width`: The width of the card(s)
  - `layout`: The layout of the card(s), e.g. 'row', 'column' or 'grid'
  - `refresh_rate`: How frequently to poll for updates in milliseconds
