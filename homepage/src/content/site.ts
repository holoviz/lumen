/**
 * All copy and link targets for the homepage, kept out of the components so wording can
 * be reviewed without reading JSX.
 *
 * Docs links are absolute under /en/docs/latest/ because the homepage is unversioned and
 * lives at the site root, while the Zensical build is published per version.
 * `homepage-serve` points them at the live site instead.
 */

import {asset, video} from './assets'

// The lumen-dev staging site has no latest build, so its homepage links into dev.
export const DOCS = import.meta.env.VITE_DOCS_PATH ?? '/en/docs/latest'
export const GITHUB = 'https://github.com/holoviz/lumen'
const DISCOURSE = 'https://discourse.holoviz.org/c/lumen/'
const DISCORD = 'https://discord.com/invite/rb6gPXbdAr'

export const nav = [
  {label: 'Docs', href: `${DOCS}/`},
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
  title: 'Illuminate your data.',
  tagline: 'Agent framework turning natural language into SQL, charts, dashboards and reports.',
  lede:
    'Ask a question in plain English and Lumen plans the steps, writes the SQL, runs it against your own sources and draws the chart, whether the data is a CSV file, a warehouse table, a NetCDF grid or a GeoJSON map. Open source, with any LLM and any database, and every step can be changed in Python or YAML.',
  install: 'pip install lumen',
  installConda: 'conda install -c conda-forge lumen',
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
 * The hero chat replays three real answers rather than calling an LLM. The SQL is what Lumen
 * wrote and the charts are the Vega-Lite specs it produced, rendered by scripts/hero_charts.py.
 */
export const heroChat = {
  dataset: 'penguins.csv',
  questions: [
    {
      prompt: 'Which islands have the most penguins?',
      chart: 'islands',
      alt: 'A bar chart of penguins per island, with Biscoe highlighted at 168.',
      sql: `SELECT
  "island",
  COUNT(*) AS penguin_count
FROM (
  SELECT DISTINCT
    *
  FROM penguins_csv
) AS deduped
GROUP BY
  "island"
ORDER BY
  penguin_count DESC`,
    },
    {
      prompt: 'Which species is the heaviest on average?',
      chart: 'mass',
      alt: 'A bar chart of average body mass by species, with Gentoo highlighted at about 5,076 grams.',
      sql: `SELECT
  "species",
  AVG(TRY_CAST("body_mass_g" AS DOUBLE)) AS avg_body_mass_g
FROM (
  SELECT DISTINCT
    *
  FROM penguins_csv
) AS "penguins_csv"
WHERE
  NOT "body_mass_g" IS NULL AND TRIM("body_mass_g") <> ''
GROUP BY
  "species"
ORDER BY
  avg_body_mass_g DESC`,
    },
    {
      prompt: 'How many of each species are there?',
      chart: 'species',
      alt: 'A bar chart of penguin counts by species: Adelie 152, Gentoo 124, Chinstrap 68.',
      sql: `SELECT
  "species",
  COUNT(*) AS "count"
FROM (
  SELECT DISTINCT
    *
  FROM penguins_csv
) AS "deduped"
GROUP BY
  "species"
ORDER BY
  "count" DESC`,
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
      body: 'The coordinator picks which of the nine built-in agents to use and in what order.',
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
      code: ["pip install 'lumen[xarray]'", 'lumen-ai serve air_temperature.nc --show'].join('\n'),
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
      code: ["pip install 'lumen[geo]'", 'lumen-ai serve countries.geojson --show'].join('\n'),
      question: 'Map each country, coloured by population.',
      sql: ['SELECT name, pop_est, geometry', 'FROM countries'].join('\n'),
      link: {label: 'Data sources', href: `${DOCS}/configuration/sources/`},
    },
  ],
}

/**
 * Every provider here is a key of LLM_PROVIDERS in lumen/ai/llm.py, the list
 * `lumen-ai serve --provider` accepts. Grouped the way docs/configuration/llm_providers.md is.
 */
export const providers = {
  title: 'Bring your own LLM',
  lede:
    'Lumen works with hosted APIs, local models, gateways and the coding CLIs you already sign in to. Lumen finds the provider from your environment, or you can name one with `--provider`. Each agent can use a different model, and Lumen tracks token usage and cost as it goes.',
  groups: [
    {
      title: 'Cloud APIs',
      names: ['OpenAI', 'Anthropic', 'Google Gemini', 'Mistral', 'Azure OpenAI', 'Azure Mistral', 'AWS Bedrock', 'Groq'],
    },
    {
      title: 'Local models',
      names: ['Ollama', 'Llama.cpp', 'MLX on Apple Silicon', 'AI Navigator'],
    },
    {
      title: 'Gateways',
      names: ['OpenRouter', 'Kilo', 'LiteLLM', 'AI Catalyst'],
    },
    {
      title: 'Coding CLIs, no API key',
      names: ['Codex CLI', 'Claude Code', 'GitHub Copilot CLI', 'Antigravity CLI'],
    },
  ],
  cli: {
    title: 'Use the subscription you already have',
    body:
      'Sign in to Codex, Claude Code, GitHub Copilot or Antigravity once, and Lumen sends its requests through that CLI. Copilot works on any plan, including Copilot Free. These providers are for local use, not shared deployments.',
    code: [
      'lumen-ai serve penguins.csv --provider claude-code',
      'lumen-ai serve penguins.csv --provider codex-cli',
      'lumen-ai serve penguins.csv --provider copilot-cli',
      'lumen-ai serve penguins.csv --provider antigravity-cli',
    ].join('\n'),
  },
  link: {label: 'All LLM providers', href: `${DOCS}/configuration/llm_providers/`},
}

/** Sources Lumen ships, from lumen/sources/ and the supported-sources table in the docs. */
export const sources = {
  title: 'Connect to the data you have',
  lede:
    'Start from a file or point Lumen at a warehouse. Mix local files and remote databases in one session.',
  items: [
    {name: 'Files', note: 'CSV, Parquet, JSON and Excel, uploaded or read from a URL'},
    {name: 'DuckDB', note: 'Fast local SQL over files and in-memory tables'},
    {name: 'PostgreSQL, MySQL and more', note: 'SQLite, Oracle and SQL Server through SQLAlchemy'},
    {name: 'Snowflake', note: 'Cloud warehouse, with SSO sign-in'},
    {name: 'BigQuery', note: "Google's data warehouse"},
    {name: 'xarray', note: 'NetCDF, Zarr, HDF5 and GRIB as SQL tables'},
    {name: 'Geospatial files', note: 'GeoJSON, WKT and zipped shapefiles'},
    {name: 'Intake catalogs', note: 'Catalogs of datasets described in YAML'},
    {name: 'Documentation', note: 'Upload files as metadata the agents read for context'},
  ],
  code: [
    'lumen-ai serve penguins.csv earthquakes.parquet',
    'lumen-ai serve postgresql://user:password@localhost:5432/mydb',
    'lumen-ai serve data.csv postgresql://localhost/mydb',
  ].join('\n'),
  link: {label: 'All data sources', href: `${DOCS}/configuration/sources/`},
}

const tutorial = (name: string) => `${DOCS}/examples/tutorials/${name}/`
const galleryPage = (name: string) => `${DOCS}/examples/gallery/${name}/`

/** Screenshots come from docs/assets, the same pictures the tutorial and gallery pages use. */
export const gallery = {
  title: 'Built with Lumen',
  lede:
    'Domain-specific AI explorers from the tutorials, and dashboards written as YAML specs. Each one comes with the full code.',
  groups: [
    {
      title: 'AI explorers',
      items: [
        {image: 'tutorials/skew_t', href: tutorial('weather_data_ai_explorer'), title: 'Weather data explorer', note: 'Custom agents that draw Skew-T diagrams from soundings'},
        {image: 'tutorials/census', href: tutorial('census_data_ai_explorer'), title: 'Census data explorer', note: 'Fetch U.S. Census data from a custom control'},
        {image: 'tutorials/mesonet', href: tutorial('mesonet_weather_explorer'), title: 'Mesonet weather explorer', note: 'Daily observations from a REST API'},
        {image: 'tutorials/massive', href: tutorial('massive_stock_explorer'), title: 'Stock market explorer', note: 'Controls generated from an SDK’s method signatures'},
      ],
    },
    {
      title: 'YAML dashboards',
      items: [
        {image: 'gallery/nyc_taxi', href: galleryPage('nyc_taxi'), title: 'NYC taxi', note: 'Million-row trip analysis'},
        {image: 'gallery/earthquakes', href: galleryPage('earthquakes'), title: 'Earthquakes', note: 'Magnitude and depth around the globe'},
        {image: 'gallery/windturbines', href: galleryPage('windturbines'), title: 'Wind turbines', note: 'Performance and location of US turbines'},
        {image: 'gallery/bikes', href: galleryPage('bikes'), title: 'London bike points', note: 'Live bike sharing on a map'},
      ],
    },
  ].map((g) => ({...g, items: g.items.map((i) => ({...i, image: asset(`${i.image}.webp`)}))})),
  link: {label: 'Browse all examples', href: `${DOCS}/examples/gallery/`},
}

/** Three ways to make Lumen yours, each from docs/index.md's "Designed for extension" tabs. */
export const extend = {
  title: 'Change any step in Python or YAML',
  lede:
    'Lumen is plain Python on top of Panel. Teach the agents your terms, give them your own functions and MCP servers, add deterministic analyses, and keep what the AI builds as a reproducible YAML spec.',
  items: [
    {
      title: 'Teach it your domain',
      body: 'Add your business rules and terminology to any agent’s prompt.',
      code: [
        'context = """',
        '{{ super() }}',
        '"Active" means logged in within 30 days.',
        'Q1 is January to March.',
        '"""',
        '',
        'agent = lmai.agents.ChatAgent(',
        '    template_overrides={"main": {"context": context}}',
        ')',
      ].join('\n'),
      link: {label: 'Prompts', href: `${DOCS}/configuration/prompts/`},
    },
    {
      title: 'Give it your own tools',
      body: 'Wrap any Python function, or connect an MCP server with `MCPTool`, and agents call it when they need it.',
      code: [
        'def customer_ltv(customer_id: str, table) -> dict:',
        '    """Calculate customer lifetime value."""',
        '    rows = table[table["customer_id"] == customer_id]',
        '    return {"ltv": rows["revenue"].sum()}',
        '',
        'tool = lmai.tools.FunctionTool(',
        '    function=customer_ltv, requires=["table"], provides=["ltv"]',
        ')',
      ].join('\n'),
      link: {label: 'Tools', href: `${DOCS}/configuration/tools/`},
    },
    {
      title: 'Keep it as a spec',
      body: 'Every dashboard is a Lumen YAML spec, so you can edit it, version it and serve it without the AI.',
      code: [
        'sources:',
        '  penguins:',
        '    type: file',
        '    tables:',
        '      penguins: penguins.csv',
        'layouts:',
        '  - title: Plots',
        '    pipeline: penguins',
        '    views:',
        '      scatter: {type: hvplot, kind: points}',
      ].join('\n'),
      link: {label: 'YAML specs', href: `${DOCS}/configuration/spec/`},
    },
  ],
}

/** Laptop to team, the one real sequence on the page. Commands from docs/configuration/cli.md and spec/deployment.md. */
export const paths = {
  title: 'From your laptop to your team',
  lede: 'Explore on your own machine, keep what works, then serve it for everyone else.',
  steps: [
    {
      title: 'Explore',
      body: 'Point `lumen-ai serve` at files, URLs or database connection strings and start asking questions in the browser.',
      code: 'lumen-ai serve data/*.csv --show',
    },
    {
      title: 'Build',
      body: 'Wrap the explorer in a Python app with your own agents, tools and context, or keep a dashboard as a YAML spec.',
      code: 'lumen serve dashboard.yaml --autoreload',
    },
    {
      title: 'Share',
      body: "Lumen apps are Panel apps, so they deploy the same way, behind Panel's authentication providers on your own infrastructure.",
      code: 'panel serve app.py --address 0.0.0.0',
    },
  ],
}

/** Features beyond a single answer, each verified against lumen/ai on main. */
export const capabilities = {
  title: 'More than one answer at a time',
  lede: 'Collect analyses into reports, run real calculations and keep an eye on what the model costs.',
  items: [
    {
      title: 'Reports and stories',
      body: 'Gather every chart and table into a report, rerun it on new data, and let Lumen write an editable story around it. Export as a notebook, HTML or Word document.',
      href: `${DOCS}/configuration/reports/`,
    },
    {
      title: 'Python calculations',
      body: 'Install the Monty integration and agents get a sandboxed `run_python` tool for exact arithmetic instead of guesses.',
      href: `${DOCS}/configuration/tools/`,
    },
    {
      title: 'Table statistics and documents',
      body: 'Lumen describes each table to the model with types, keys, value ranges and frequent values. Upload documentation as metadata and vector stores retrieve the relevant parts for each question.',
      href: `${DOCS}/configuration/vector_stores/`,
    },
    {
      title: 'Token usage and cost',
      body: 'Every LLM records the tokens it uses. Add your per-million-token prices and Lumen estimates the cost of each request.',
      href: `${DOCS}/configuration/llm_providers/#track-token-usage-and-cost`,
    },
  ],
}

export const adoption = {
  title: 'Open source, part of HoloViz',
  body:
    'Lumen is built on Panel, hvPlot and the rest of the HoloViz tools and developed in the open. Ask questions on Discourse or Discord, and report bugs on GitHub.',
  // Checked against the GitHub API and PyPI on 2026-10-14; nothing here reads them live.
  stats: [
    {value: '1.4', label: 'Latest release'},
    {value: '2020', label: 'First release'},
    {value: '40+', label: 'Contributors'},
    {value: 'BSD-3', label: 'License'},
  ],
}

/**
 * Screenshots of real Lumen answers, taken by scripts/showcase.py against a live server. The
 * caption is the exact question that produced each one.
 */
export const showcase = {
  title: 'Real answers, straight from the app',
  lede:
    'Everything here is a real question typed into Lumen and the answer it drew: earthquakes binned into 3D hexagons, the busiest US flight routes as arcs, three linked charts from one request, a climate grid from a NetCDF file and world cities as towers.',
  items: [
    {
      name: 'geo-video',
      title: 'Watch: maps in 3D',
      caption: 'Earthquakes stacked into 3D towers, then the busiest US flight routes drawn as arcs, from two questions in plain English.',
      video: true,
    },
    {
      name: 'charts-video',
      title: 'Watch: charts and analysis',
      caption: 'Ask for a map and a trend line of every earthquake, and Lumen draws both and explains what they show.',
      video: true,
    },
    {
      name: 'penguins',
      title: 'Three charts from one request',
      caption: 'Three linked charts of penguins from a single question: body mass, flipper length and where they live.',
    },
    {
      name: 'heatmap',
      title: 'A NetCDF climate grid',
      caption: 'Two years of air temperature over North America as a colour map, from a scientific data file.',
    },
    {
      name: 'cities',
      title: 'Cities as towers',
      caption: 'The world\'s cities as towers, taller for bigger populations.',
    },
  ].map((s) =>
    s.video
      ? {...s, image: asset(`showcase/${s.name}-poster.webp`), clip: video(`showcase/${s.name}`)}
      : {...s, image: asset(`showcase/${s.name}.webp`), clip: undefined},
  ),
}
