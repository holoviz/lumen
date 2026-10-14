/**
 * All copy and link targets for the homepage, kept out of the components so wording can
 * be reviewed without reading JSX.
 *
 * The homepage is copied over the root of the Zensical build, so docs links are root
 * relative directory URLs. `homepage-serve` points them at the live site instead.
 */

export const DOCS = import.meta.env.VITE_DOCS_PATH ?? ''
export const GITHUB = 'https://github.com/holoviz/lumen'
const DISCOURSE = 'https://discourse.holoviz.org/c/lumen/'
const DISCORD = 'https://discord.com/invite/rb6gPXbdAr'

export const nav = [
  {label: 'Docs', href: `${DOCS}/overview/`},
  {label: 'Quick Start', href: `${DOCS}/quick_start/`},
  {label: 'Tutorials', href: `${DOCS}/examples/tutorials/`},
  {label: 'Gallery', href: `${DOCS}/examples/gallery/`},
  {label: 'API', href: `${DOCS}/reference/api/`},
]

export const getStarted = {label: 'Get started', href: `${DOCS}/quick_start/`}

export const social = [
  {label: 'GitHub', href: GITHUB},
  {label: 'Discord', href: DISCORD},
  {label: 'Discourse', href: DISCOURSE},
]

export const footerColumns = [
  {
    heading: 'Learn',
    links: [
      {label: 'Quick start', href: `${DOCS}/quick_start/`},
      {label: 'Installation', href: `${DOCS}/installation/`},
      {label: 'Getting started', href: `${DOCS}/getting_started/launching_lumen/`},
      {label: 'Tutorials', href: `${DOCS}/examples/tutorials/`},
    ],
  },
  {
    heading: 'Reference',
    links: [
      {label: 'Configuration', href: `${DOCS}/configuration/cli/`},
      {label: 'API reference', href: `${DOCS}/reference/api/`},
      {label: 'Gallery', href: `${DOCS}/examples/gallery/`},
      {label: 'Release notes', href: `${DOCS}/releases/`},
    ],
  },
  {
    heading: 'Community',
    links: [
      {label: 'Discourse', href: DISCOURSE},
      {label: 'Discord', href: DISCORD},
      {label: 'GitHub issues', href: `${GITHUB}/issues`},
      {label: 'Contributing', href: `${DOCS}/contributing/`},
    ],
  },
  {
    heading: 'HoloViz',
    links: [
      {label: 'Panel', href: 'https://panel.holoviz.org'},
      {label: 'hvPlot', href: 'https://hvplot.holoviz.org'},
      {label: 'HoloViews', href: 'https://holoviews.org'},
      {label: 'Param', href: 'https://param.holoviz.org'},
    ],
  },
]

export const hero = {
  title: 'Chat with your data.',
  lede:
    'Lumen is an open-source Python framework for AI data applications. Ask a question in plain English and Lumen plans the steps, writes the SQL, runs it against your own sources and draws the chart, whether the data is a CSV file, a warehouse table, a NetCDF grid or a GeoJSON map. Use any LLM, connect any database, and change every step in Python or YAML.',
  install: "pip install 'lumen[ai-openai]'",
  installConda: 'conda install -c conda-forge lumen openai',
  primary: getStarted,
  secondary: {label: 'See examples', href: `${DOCS}/examples/gallery/`},
}

/** The two ways to start the app shown above it, both from the quick start. */
export const heroCode = [
  {
    label: 'Point it at a file or database.',
    source: 'lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv --show',
  },
  {
    label: 'Or start it from Python.',
    source: [
      'from lumen.ai import ExplorerUI',
      '',
      "ui = ExplorerUI(data='https://datasets.holoviz.org/penguins/v1/penguins.csv')",
      'ui.servable()',
    ].join('\n'),
  },
]

/**
 * The hero chat replays three real answers rather than calling an LLM. Every number is the
 * result of the SQL beside it on the penguins dataset the quick start uses (344 rows).
 */
export const heroChat = {
  dataset: 'penguins.csv',
  questions: [
    {
      prompt: 'Which islands have the most penguins?',
      sql: ['SELECT island, COUNT(*) AS penguins', 'FROM penguins', 'GROUP BY island', 'ORDER BY penguins DESC'].join('\n'),
      title: 'Penguins by island',
      bars: [
        {label: 'Biscoe', value: 168},
        {label: 'Dream', value: 124},
        {label: 'Torgersen', value: 52},
      ],
    },
    {
      prompt: 'Which species is the heaviest on average?',
      sql: [
        'SELECT species, AVG(body_mass_g) AS body_mass',
        'FROM penguins',
        'GROUP BY species',
        'ORDER BY body_mass DESC',
      ].join('\n'),
      title: 'Average body mass (g)',
      bars: [
        {label: 'Gentoo', value: 5076},
        {label: 'Chinstrap', value: 3733},
        {label: 'Adelie', value: 3701},
      ],
    },
    {
      prompt: 'How many of each species are there?',
      sql: ['SELECT species, COUNT(*) AS penguins', 'FROM penguins', 'GROUP BY species', 'ORDER BY penguins DESC'].join('\n'),
      title: 'Penguins by species',
      bars: [
        {label: 'Adelie', value: 152},
        {label: 'Gentoo', value: 124},
        {label: 'Chinstrap', value: 68},
      ],
    },
  ],
  /** The plan Lumen shows for a charting question: query, chart, then check the answer. */
  steps: [
    {agent: 'SQLAgent', text: 'Query the table'},
    {agent: 'VegaLiteAgent', text: 'Chart the result'},
    {agent: 'ValidationAgent', text: 'Check the question was answered'},
  ],
}

/** One question, four steps. Each step names the part of Lumen that does it. */
export const howItWorks = {
  title: 'From a question to a chart you can edit',
  lede:
    'Lumen splits every question into steps and gives each step to an agent built for it. You see the plan, the SQL and the chart spec, and you can change any of them by hand or by asking again.',
  steps: [
    {
      title: 'Ask',
      body: 'Type a question in plain English, or upload a file and ask about it.',
      code: 'Which islands have the most penguins?',
    },
    {
      title: 'Plan',
      body: 'The coordinator picks which of the eight built-in agents to use and in what order.',
      code: 'Planner',
    },
    {
      title: 'Query',
      body: 'SQLAgent writes SQL for your source and runs it there, so the work happens in your database.',
      code: 'SQLAgent',
    },
    {
      title: 'Answer',
      body: 'VegaLiteAgent draws the chart and ValidationAgent checks it answers the question. SQL and chart specs open in an editor.',
      code: 'VegaLiteAgent',
    },
  ],
  link: {label: 'How coordinators and agents work', href: `${DOCS}/configuration/coordinators/`},
}

/**
 * Lumen is not limited to flat tables: XArraySQLSource exposes xarray datasets as SQL tables,
 * and geometry columns flow through to map charts.
 */
export const beyondTables = {
  title: 'Query grids and maps, not only tables',
  lede:
    'Scientific and geospatial data work the same way as a spreadsheet. Ask about temperature across time, latitude and longitude, or about shapes on a map, and Lumen writes the query and draws the result.',
  items: [
    {
      title: 'N-dimensional arrays',
      body:
        'Open NetCDF, Zarr, HDF5 and GRIB files lazily with xarray. Each variable becomes a SQL table with its dimensions as columns, so a question can slice, group and average across time, latitude, longitude or any other dimension. Results plot as grids, maps or lines.',
      formats: ['NetCDF', 'Zarr', 'HDF5', 'GRIB'],
      code: ["pip install 'lumen[ai-openai,xarray]'", 'lumen-ai serve air_temperature.nc --show'].join('\n'),
      question: 'What was the mean air temperature over North America each month of 2013?',
      sql: [
        "SELECT date_trunc('month', time) AS month, AVG(air) AS mean_air",
        'FROM air',
        'WHERE lat BETWEEN 15 AND 75 AND lon BETWEEN 200 AND 310',
        "  AND time < '2014-01-01'",
        'GROUP BY month ORDER BY month',
      ].join('\n'),
      link: {label: 'Scientific data sources', href: `${DOCS}/configuration/sources/#xarray-for-scientific-data`},
    },
    {
      title: 'Geospatial data',
      body:
        'Load GeoJSON, WKT and zipped shapefiles, or any GeoDataFrame. Geometry columns keep their coordinate reference system and render as maps with Vega-Lite or Deck.gl, so a question about places comes back as a map.',
      formats: ['GeoJSON', 'WKT', 'Shapefile', 'GeoDataFrame'],
      code: ["pip install 'lumen[ai-openai,geo]'", 'lumen-ai serve countries.geojson --show'].join('\n'),
      question: 'Map each country, coloured by population.',
      sql: ['SELECT name, pop_est, geometry', 'FROM countries'].join('\n'),
      link: {label: 'Data sources', href: `${DOCS}/configuration/sources/`},
    },
  ],
}
