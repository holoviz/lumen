# Dashboard specification

The Lumen dashboard can be configured by a `dashboard.yml` file. The dashboard yaml follows the following specification:

- `config`:
  - `title`: The title of the overall application
  - `layout`: The layout to put the targets in
  - `logo`: A URL or local path to an image file
  - `template`: The template to use for the monitoring application
  - `ncols`: The number of columns to use in the grid of cards
- `targets`: This is the list of targets to monitor
  - `title`: The title of the monitoring endpoint
    `source`: The `Source` used to monitor an endpoint
      `type`: The type of `Source` to use, e.g. 'rest' or 'live'
	  `...`: Additional parameters for the `Source`
    `views`: A list of metrics to monitor and display on the endpoint
      - `table`: The name of the table to visualize
	    `type`: The type of `View` to use for rendering the table
	    `...`: Additional parameters for the `View`
    `filters`: A list of `Filter` types to select a subset of the data
      - `field`: The name of the filter
	    `type`: The type of the `Filter` to use, e.g. 'constant', 'widget' or 'facet'
	    `...`: Additional parameters for the `Filter`
    `layout`: The layout of the card(s), e.g. 'row', 'column' or 'grid'
    `refresh_rate`: How frequently to poll for updates in milliseconds
	`...`: Additional parameters passed to the `Card` layout(s), e.g. `width` or `height`
