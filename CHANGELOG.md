# Releases

## Version 1.4.0

Lumen AI 1.4 makes Lumen easier to run anywhere and cheaper to run well. It adds keyless and CLI-backed LLM providers, tracks token usage and cost, gives agents more accurate table context, and lets reports carry an AI-written story that exports to Word. It also adds polars and pyarrow support through narwhals, broader geospatial rendering, and a large batch of reliability fixes for SQL, validation, and charts.

### LLM Providers & Models

* Kilo provider, including a keyless free tier that works end to end ([#1984](https://github.com/holoviz/lumen/pull/1984), [#2198](https://github.com/holoviz/lumen/pull/2198))
* Codex CLI, Claude Code, and GitHub Copilot CLI providers ([#2070](https://github.com/holoviz/lumen/pull/2070), [#2121](https://github.com/holoviz/lumen/pull/2121))
* Guide users without an API key to a working provider on first run ([#2198](https://github.com/holoviz/lumen/pull/2198))
* Opt-in model routing ([#2015](https://github.com/holoviz/lumen/pull/2015))
* `DecisionModel` interface with Jev and OpenRouter providers for closed-choice decisions ([#2109](https://github.com/holoviz/lumen/pull/2109))
* Make `gpt-5.6-luna` the default OpenAI model and refresh provider model lists ([#2007](https://github.com/holoviz/lumen/pull/2007), [#2118](https://github.com/holoviz/lumen/pull/2118), [#2182](https://github.com/holoviz/lumen/pull/2182))
* Read `OLLAMA_HOST` and default local providers to a 9B model ([#2202](https://github.com/holoviz/lumen/pull/2202))
* `--llm-model-url` now loads the GGUF model via llama-cpp ([#2201](https://github.com/holoviz/lumen/pull/2201))
* Fix Anthropic structured output for modern Claude models ([#2196](https://github.com/holoviz/lumen/pull/2196))
* Run Claude Code CLI requests without its built-in tools, MCP servers or user settings, in a fresh directory ([#2214](https://github.com/holoviz/lumen/pull/2214))
* Show which model answered in the chat message footer ([#2053](https://github.com/holoviz/lumen/pull/2053))
* Run Claude Code CLI requests without plan mode ([#2229](https://github.com/holoviz/lumen/pull/2229))
* Fall back to JSON mode when Kilo's upstream rejects structured requests ([#2234](https://github.com/holoviz/lumen/pull/2234))
* Use `json_schema` mode for Groq, so server-side tool validation no longer rejects every chart ([#2239](https://github.com/holoviz/lumen/pull/2239))

### Agents, Tools & Observability

* Track LLM token usage and estimated costs ([#2100](https://github.com/holoviz/lumen/pull/2100))
* Capture scoped LLM model and tool traces, including streamed usage ([#2101](https://github.com/holoviz/lumen/pull/2101), [#2102](https://github.com/holoviz/lumen/pull/2102))
* Accurate, compact, and cache-friendly table context for AI agents ([#2105](https://github.com/holoviz/lumen/pull/2105), [#2116](https://github.com/holoviz/lumen/pull/2116))
* Return tool errors to the model and let SQLAgent answer inside the tool loop ([#2104](https://github.com/holoviz/lumen/pull/2104), [#2117](https://github.com/holoviz/lumen/pull/2117))
* Optional Monty Python sandbox tool for agents ([#2098](https://github.com/holoviz/lumen/pull/2098))
* Data cleaning mode for SQLAgent ([#2000](https://github.com/holoviz/lumen/pull/2000))
* Few-shot examples for the Planner ([#2094](https://github.com/holoviz/lumen/pull/2094))
* `run_in_thread` option for sync `FunctionTool` functions, plus fixes for multiple `provides` keys and `step_title` forwarding ([#2138](https://github.com/holoviz/lumen/pull/2138), [#2137](https://github.com/holoviz/lumen/pull/2137), [#2135](https://github.com/holoviz/lumen/pull/2135))
* Shorter prompts and lower token usage, with tweaks that help small models ([#1986](https://github.com/holoviz/lumen/pull/1986), [#1989](https://github.com/holoviz/lumen/pull/1989), [#1991](https://github.com/holoviz/lumen/pull/1991))
* Unified agent names and avatars, and a polished planner checklist ([#2096](https://github.com/holoviz/lumen/pull/2096), [#2115](https://github.com/holoviz/lumen/pull/2115))
* Column-level lineage tooltips: hover a derived column in a SQL result to see its expression and grouping ([#2124](https://github.com/holoviz/lumen/pull/2124))

### Validation & Reliability

* Fix deterministic AI failures found by the evals ([#2103](https://github.com/holoviz/lumen/pull/2103))
* ValidationAgent no longer fails valid plans on stale context, with a rule-based heuristic for previous keys ([#2074](https://github.com/holoviz/lumen/pull/2074), [#2081](https://github.com/holoviz/lumen/pull/2081), [#2065](https://github.com/holoviz/lumen/pull/2065))
* Recover from unparsable SQL in `SQLAgent.revise()` instead of crashing ([#2054](https://github.com/holoviz/lumen/pull/2054))
* Only flag placeholder numbers outside a column's value range ([#2145](https://github.com/holoviz/lumen/pull/2145))
* Quote bare table names that sqlglot mis-parses ([#1995](https://github.com/holoviz/lumen/pull/1995))
* `ViewOutputs.view` is now a required output key ([#2108](https://github.com/holoviz/lumen/pull/2108))
* Judge duplicate rows on whole rows, so the cleaning step no longer deduplicates records that only share the columns a query reads ([#2235](https://github.com/holoviz/lumen/pull/2235))
* Skip the validation step for an empty plan ([#2236](https://github.com/holoviz/lumen/pull/2236))
* Fix `Filter` warning formatting and call the existing `_embed` instead of the undefined `_embed_actions` ([#2018](https://github.com/holoviz/lumen/pull/2018), [#2027](https://github.com/holoviz/lumen/pull/2027))

### Reports

* Annotate reports with an AI-written story, editable with Panel's `TextEditor` ([#1923](https://github.com/holoviz/lumen/pull/1923), [#2017](https://github.com/holoviz/lumen/pull/2017))
* Export a report as Word (`.docx`) ([#1924](https://github.com/holoviz/lumen/pull/1924))

### Data Sources & Performance

* Support polars and pyarrow frames via narwhals ([#2008](https://github.com/holoviz/lumen/pull/2008), [#2039](https://github.com/holoviz/lumen/pull/2039), [#2059](https://github.com/holoviz/lumen/pull/2059))
* Narrow queries to the columns the transforms need ([#2051](https://github.com/holoviz/lumen/pull/2051))
* Preview exploration results and build AI summaries without converting the whole frame ([#2058](https://github.com/holoviz/lumen/pull/2058), [#2044](https://github.com/holoviz/lumen/pull/2044))
* Keep integer and boolean dtypes across the pandas fallback ([#2046](https://github.com/holoviz/lumen/pull/2046))
* Skip row counts when generating auto filters and merging `JoinedSource` schemas ([#2140](https://github.com/holoviz/lumen/pull/2140), [#2119](https://github.com/holoviz/lumen/pull/2119))
* Support open-ended ranges and slices, and NumPy scalar values, in `SQLFilter` and `Filter` ([#2114](https://github.com/holoviz/lumen/pull/2114), [#2125](https://github.com/holoviz/lumen/pull/2125))
* Skip `information_schema` when listing `SQLAlchemySource` tables ([#2142](https://github.com/holoviz/lumen/pull/2142))
* Apply `Source.metadata` descriptions in `get_metadata` ([#2180](https://github.com/holoviz/lumen/pull/2180))
* Keep source actions defined with postponed annotations ([#2141](https://github.com/holoviz/lumen/pull/2141))
* Fix three blockers for local and file-based sources, and handle non-default catalog indexes ([#1993](https://github.com/holoviz/lumen/pull/1993), [#2029](https://github.com/holoviz/lumen/pull/2029))
* Read table functions with keyword arguments, such as `read_csv(..., nullstr='NA')` ([#2233](https://github.com/holoviz/lumen/pull/2233))
* Avoid passing `how` together with `thresh` in `DropNA` ([#2021](https://github.com/holoviz/lumen/pull/2021))

### Geospatial & Scientific Data

* Render a table's own geometry as a Vega-Lite geoshape ([#1998](https://github.com/holoviz/lumen/pull/1998))
* Load GeoJSON uploads end to end and preserve GeoDataFrame CRS at DuckDB ingest ([#1996](https://github.com/holoviz/lumen/pull/1996), [#2077](https://github.com/holoviz/lumen/pull/2077))
* Teach the DeckGL prompt `GeoJsonLayer` and data-driven colors, and serialize datetime columns in GeoJSON layers ([#1999](https://github.com/holoviz/lumen/pull/1999), [#1997](https://github.com/holoviz/lumen/pull/1997))
* Feed the table to basemap point layers and fix blank choropleths for unregistered lookups ([#2151](https://github.com/holoviz/lumen/pull/2151), [#2001](https://github.com/holoviz/lumen/pull/2001))
* Collapse cftime gridded dims on xarray-sql 0.6, and decline to grid a `to_dataset` result that is not a grid ([#2144](https://github.com/holoviz/lumen/pull/2144), [#2049](https://github.com/holoviz/lumen/pull/2049))
* Render geometry in Vega-Lite views, both from a table's own geometry and when a spec joins external boundaries ([#2232](https://github.com/holoviz/lumen/pull/2232))

### Charts & Visualization

* Improved VegaLiteAgent ([#1992](https://github.com/holoviz/lumen/pull/1992))
* Datashader categorical blending on hvPlot views ([#2035](https://github.com/holoviz/lumen/pull/2035))
* Allow all tabular hvPlot kinds, and let the hvPlot agent generate a plot at all ([#2033](https://github.com/holoviz/lumen/pull/2033), [#2064](https://github.com/holoviz/lumen/pull/2064))
* Draw hvPlot glyphs in the chat UI, not just the axes ([#2071](https://github.com/holoviz/lumen/pull/2071))
* Coerce date fields to temporal encoding in `VegaLiteView` ([#2055](https://github.com/holoviz/lumen/pull/2055))
* Drop the unconditional multi-series split rule from the Vega-Lite prompt ([#2076](https://github.com/holoviz/lumen/pull/2076))
* Thin crowded heatmap axis labels, stop scientific notation in labels, and rewrite comma number formats ([#2148](https://github.com/holoviz/lumen/pull/2148), [#2154](https://github.com/holoviz/lumen/pull/2154), [#2188](https://github.com/holoviz/lumen/pull/2188))
* Stop awaiting the async generator that streams view specs ([#2034](https://github.com/holoviz/lumen/pull/2034))
* Escape dotted column names such as `nino3.4_anom` in Vega-Lite specs, which drew no series or a blank chart ([#2215](https://github.com/holoviz/lumen/pull/2215))

### UX & CLI

* "Follow system" option for the light and dark theme toggle ([#1970](https://github.com/holoviz/lumen/pull/1970))
* Fix the Full data toggle overlap in explorations and let chat steps shrink with the column ([#2045](https://github.com/holoviz/lumen/pull/2045), [#2152](https://github.com/holoviz/lumen/pull/2152))
* Fix `lumen serve <spec>.yaml` routing to the AI handler ([#2075](https://github.com/holoviz/lumen/pull/2075))
* Correct the CLI version output and code execution ([#2166](https://github.com/holoviz/lumen/pull/2166))
* Keep each exploration's chat when switching explorations while a plan runs ([#2213](https://github.com/holoviz/lumen/pull/2213))
* Center planner checklist status icons on the label's first line ([#2217](https://github.com/holoviz/lumen/pull/2217))
* Keep the nav drawer toggle clear of chat avatars ([#2218](https://github.com/holoviz/lumen/pull/2218))
* Size the code editor to match the body text ([#2219](https://github.com/holoviz/lumen/pull/2219))
* Show agent avatars while a response is streaming ([#2216](https://github.com/holoviz/lumen/pull/2216))
* Add a homepage for Lumen ([#2189](https://github.com/holoviz/lumen/pull/2189))

### Deprecations

* `HistoryTransform` is deprecated ([#2060](https://github.com/holoviz/lumen/pull/2060))

### Dependencies & Compatibility

* Require panel 1.9.0+ and bokeh 3.9.0+, pin griffe below 2.0, and exclude param 2.5.0a0 ([#2079](https://github.com/holoviz/lumen/pull/2079), [#2087](https://github.com/holoviz/lumen/pull/2087))
* Require pillow 13.0.0+ ([#2240](https://github.com/holoviz/lumen/pull/2240))
* Exclude litellm 1.107.0, which was compromised on PyPI ([#2205](https://github.com/holoviz/lumen/pull/2205))
* Declare all top-level imports as direct dependencies and support metadata 2.5 in PyPI publishing ([#2092](https://github.com/holoviz/lumen/pull/2092), [#2095](https://github.com/holoviz/lumen/pull/2095))
* Test the full range of supported Python versions ([#2093](https://github.com/holoviz/lumen/pull/2093))

### Documentation

* A docs truth pass: install and setup steps, CLI providers and flags, model keys (Groq, Kilo), every agent and export format, geospatial files and GRIB, ChromaDB, missing API reference pages, cost-friendly example models, and fixed links ([#2158](https://github.com/holoviz/lumen/pull/2158), [#2160](https://github.com/holoviz/lumen/pull/2160), [#2162](https://github.com/holoviz/lumen/pull/2162), [#2164](https://github.com/holoviz/lumen/pull/2164), [#2168](https://github.com/holoviz/lumen/pull/2168), [#2170](https://github.com/holoviz/lumen/pull/2170), [#2172](https://github.com/holoviz/lumen/pull/2172), [#2176](https://github.com/holoviz/lumen/pull/2176), [#2178](https://github.com/holoviz/lumen/pull/2178), [#2186](https://github.com/holoviz/lumen/pull/2186), [#2191](https://github.com/holoviz/lumen/pull/2191), [#2203](https://github.com/holoviz/lumen/pull/2203), [#2037](https://github.com/holoviz/lumen/pull/2037))
* Document table discovery in the embeddings guide, update the ingestion controls docs and add the spec tutorial to the navigation ([#2019](https://github.com/holoviz/lumen/pull/2019), [#2022](https://github.com/holoviz/lumen/pull/2022), [#2030](https://github.com/holoviz/lumen/pull/2030))

### Contributors

Many thanks to @ghostiee-11, @philippjfr, @ahuang11, @vignesh-chaturvedi, @SajalDevX, @DevForge255, @7487, @rimsha152007-dotcom, @fly1d, @Vamshi1404, @hoxbro, @AmirF194, @yiboban, @raashish1601, @nightcityblade, @mayuriphad, @mahdibani, @latent-9, @Zayn329, @SuMayaBee, @Mr-Neutr0n, @Coderambling, @umi008, and @jadhavgaurav for their contributions to this release.

A special welcome to our new contributors: @vignesh-chaturvedi, @SajalDevX, @DevForge255, @7487, @rimsha152007-dotcom, @fly1d, @Vamshi1404, @AmirF194, @yiboban, @raashish1601, @nightcityblade, @mayuriphad, @mahdibani, @latent-9, @Zayn329, @Mr-Neutr0n, @umi008, and @jadhavgaurav.

## Version 1.3.0

Lumen AI 1.3 expands support for geospatial and scientific data, makes AI-generated visualizations more capable, and improves the reliability of exploration, validation, reporting, and exports. This release also brings a number of UX improvements and makes Lumen more robust when working with optional dependencies and real-world datasets.

### Geospatial & Scientific Data

* End-to-end geometry column support, allowing GeoDataFrames to render as maps ([#1903](https://github.com/holoviz/lumen/pull/1903))
* Support for loading geospatial files (`.geojson`, `.wkt`, `.shp`) at startup ([#1905](https://github.com/holoviz/lumen/pull/1905))
* Coordinate-aware filters for exploration outputs ([#1894](https://github.com/holoviz/lumen/pull/1894))
* Support for plotting XArray grids with hvPlot, Vega-Lite, and DeckGL ([#1823](https://github.com/holoviz/lumen/pull/1823))
* Improved gridded plotting for real-world climate NetCDF data, including cftime and bounds variables ([#1943](https://github.com/holoviz/lumen/pull/1943), [#1944](https://github.com/holoviz/lumen/pull/1944))

### AI & Visualization

* Generate multiple charts from a single query with VegaLiteAgent ([#1927](https://github.com/holoviz/lumen/pull/1927))
* Shared categorical palettes across generated charts for more consistent visualizations ([#1936](https://github.com/holoviz/lumen/pull/1936))
* Improved Vega-Lite generation for diverging color schemes using `domainMid` ([#1933](https://github.com/holoviz/lumen/pull/1933))
* Support polygons and paths in hvPlot-generated views ([#1934](https://github.com/holoviz/lumen/pull/1934))
* Reusable Vega-Lite spec normalization utility ([#1909](https://github.com/holoviz/lumen/pull/1909))
* Improved handling of Vega-Lite specifications without schemas ([#1959](https://github.com/holoviz/lumen/pull/1959))

### Reports & Exports

* Build Reports directly from prebuilt views with `Report.from_views` ([#1910](https://github.com/holoviz/lumen/pull/1910))
* Select which report sections to retain when exporting ([#1922](https://github.com/holoviz/lumen/pull/1922))
* Add single-chart export formats to the multichart “All” tab ([#1967](https://github.com/holoviz/lumen/pull/1967))

### Agents, SQL & Reliability

* Improved validation and recovery across the AI workflow ([#1921](https://github.com/holoviz/lumen/pull/1921), [#1963](https://github.com/holoviz/lumen/pull/1963))
* View YAML auto-retry can now successfully recover from errors ([#1929](https://github.com/holoviz/lumen/pull/1929))
* Dynamically provide context to Analysis components ([#1925](https://github.com/holoviz/lumen/pull/1925))
* Automatically resolve the data source when only one source is available ([#1964](https://github.com/holoviz/lumen/pull/1964))
* Batch SQL queries into a single call ([#1977](https://github.com/holoviz/lumen/pull/1977))
* Scope schema context in revision prompts to the current table ([#1928](https://github.com/holoviz/lumen/pull/1928))
* Fix SQL subquery handling ([#1874](https://github.com/holoviz/lumen/pull/1874))
* Prevent repeated tables and collapse duplicate column names in generated results ([#1975](https://github.com/holoviz/lumen/pull/1975), [#1962](https://github.com/holoviz/lumen/pull/1962))

### Data Ingestion & Dependencies

* Detect file encoding when reading uploaded CSV, JSON, and HTML files ([#1902](https://github.com/holoviz/lumen/pull/1902))
* Added generic `try_import` helper for optional dependencies ([#1926](https://github.com/holoviz/lumen/pull/1926))
* Added `try_import_xarray` helper for optional XArray support ([#1908](https://github.com/holoviz/lumen/pull/1908))

### UX & Interface

* Improved editor layout and filtering ([#1918](https://github.com/holoviz/lumen/pull/1918))
* Improved data description and validation flows ([#1919](https://github.com/holoviz/lumen/pull/1919), [#1921](https://github.com/holoviz/lumen/pull/1921))
* Improved navigation layout and tab handling ([#1917](https://github.com/holoviz/lumen/pull/1917), [#1920](https://github.com/holoviz/lumen/pull/1920))
* Added default titles and improved LLM labeling in the AI UI ([#1973](https://github.com/holoviz/lumen/pull/1973), [#1961](https://github.com/holoviz/lumen/pull/1961))
* Fixed conflicting sizing definitions in the AI UI ([#1972](https://github.com/holoviz/lumen/pull/1972))
* Restored gallery and README images following the documentation migration ([#1955](https://github.com/holoviz/lumen/pull/1955))

### Documentation

* Fixed broken links across the contributing, views, analyses, and app-building documentation ([#1949](https://github.com/holoviz/lumen/pull/1949), [#1950](https://github.com/holoviz/lumen/pull/1950), [#1951](https://github.com/holoviz/lumen/pull/1951), [#1952](https://github.com/holoviz/lumen/pull/1952))
* Removed stale documentation references and corrected the PyPI package description ([#1969](https://github.com/holoviz/lumen/pull/1969), [#1938](https://github.com/holoviz/lumen/pull/1938))

### Contributors

Many thanks to @ahuang11, @ghostiee-11, @philippjfr, @ciaograsso06, @Sanjays2402, @Coderambling, @Hillindigo, and @w3lld1 for their contributions to this release.

A special welcome to our new contributors: @Coderambling, @Hillindigo, @Sanjays2402, and @w3lld1.

**Full changelog:** [v1.2.1 → v1.3.0](https://github.com/holoviz/lumen/compare/v1.2.1...v1.3.0)

## Version 1.2.0

Lumen AI 1.2 deepens the tool-calling architecture introduced in 1.1, adds new LLM providers and data sources, and delivers a large batch of agent reliability and UX improvements.

### New Providers & Models

* OpenRouter LLM provider integration ([#1863](https://github.com/holoviz/lumen/pull/1863))
* MLX (Apple Silicon) model support ([#1861](https://github.com/holoviz/lumen/pull/1861))
* OpenAI Responses API support ([#1807](https://github.com/holoviz/lumen/pull/1807))

### Data Sources & Ingestion

* XArraySQLSource for natural language queries on N-dimensional scientific data ([#1741](https://github.com/holoviz/lumen/pull/1741), [#1791](https://github.com/holoviz/lumen/pull/1791))
* Kaggle dataset support in DownloadSourceControls ([#1848](https://github.com/holoviz/lumen/pull/1848))
* Auto catalog source discovery ([#1826](https://github.com/holoviz/lumen/pull/1826))
* Read tables and text directly from URLs ([#1839](https://github.com/holoviz/lumen/pull/1839))
* Image upload support ([#1802](https://github.com/holoviz/lumen/pull/1802))

### Agents & Planning

* New SourceAgent for data source management ([#1810](https://github.com/holoviz/lumen/pull/1810))
* SQLAgent discovery mode replaced with tool calling ([#1803](https://github.com/holoviz/lumen/pull/1803))
* Improved Planner validation flow with clarification LLM tool ([#1833](https://github.com/holoviz/lumen/pull/1833))
* Simplified Planner reasoning and clarification ([#1836](https://github.com/holoviz/lumen/pull/1836))
* Planned context added to ChatAgent's prompt ([#1847](https://github.com/holoviz/lumen/pull/1847))
* ValidationAgent improvements: better context, logic, and TableListAgent interop ([#1820](https://github.com/holoviz/lumen/pull/1820), [#1827](https://github.com/holoviz/lumen/pull/1827), [#1845](https://github.com/holoviz/lumen/pull/1845))
* Custom clarification support ([#1856](https://github.com/holoviz/lumen/pull/1856))

### Reports & Exports

* HTML export option for Reports ([#1798](https://github.com/holoviz/lumen/pull/1798))
* Cancel support for running reports ([#1819](https://github.com/holoviz/lumen/pull/1819))
* Explain query icon in SQL editor ([#1785](https://github.com/holoviz/lumen/pull/1785))
* Task previews in the UI ([#1792](https://github.com/holoviz/lumen/pull/1792))

### UX & Interface

* Updated to use panel-material-ui ([#1876](https://github.com/holoviz/lumen/pull/1876))
* All new controls interface ([#1805](https://github.com/holoviz/lumen/pull/1805))
* Show table names in exploration tool ([#1854](https://github.com/holoviz/lumen/pull/1854))
* Show popout only when more than one exploration exists ([#1869](https://github.com/holoviz/lumen/pull/1869))
* Unified LLM message logging ([#1841](https://github.com/holoviz/lumen/pull/1841))
* Various minor UI polish fixes ([#1837](https://github.com/holoviz/lumen/pull/1837))
* Improved DocumentEditor document rendering ([#1817](https://github.com/holoviz/lumen/pull/1817))
* Downloaded source propagated into global context ([#1858](https://github.com/holoviz/lumen/pull/1858))

### Tools & Vector Stores

* LLM tools can now be provided per Actor ([#1808](https://github.com/holoviz/lumen/pull/1808))
* Custom embedding functions in ChromaDBVectorStore ([#1814](https://github.com/holoviz/lumen/pull/1814))
* DuckDBVectorStore metadata property implemented ([#1838](https://github.com/holoviz/lumen/pull/1838))
* Document search tools added ([#1813](https://github.com/holoviz/lumen/pull/1813))

### Bug Fixes

* Fix XArraySQLSource crashes on NetCDF files with bounds variables ([#1886](https://github.com/holoviz/lumen/pull/1886))
* Fix SQLAgent crash on columns absent from the source catalog ([#1881](https://github.com/holoviz/lumen/pull/1881))
* Fix SQLAgent crash on sources with the legacy 'any' dialect ([#1879](https://github.com/holoviz/lumen/pull/1879))
* Fix message format and tool handling for Google LLMs ([#1877](https://github.com/holoviz/lumen/pull/1877))
* Fix correctly logging tool call results ([#1884](https://github.com/holoviz/lumen/pull/1884))
* Ensure OpenAI LLM doesn't error when no tools provided ([#1883](https://github.com/holoviz/lumen/pull/1883))
* Fix report export ([#1870](https://github.com/holoviz/lumen/pull/1870))
* Fix `max_retries` leaking to bare SDK client in `_run_tool_loop` ([#1850](https://github.com/holoviz/lumen/pull/1850))
* Fix issues handling tool calls in OpenAI responses API ([#1811](https://github.com/holoviz/lumen/pull/1811))
* Ensure stream responses handled correctly with tool calls ([#1832](https://github.com/holoviz/lumen/pull/1832))
* Fix ingestion of source not showing in controls ([#1853](https://github.com/holoviz/lumen/pull/1853))
* Fix blocking calls ([#1818](https://github.com/holoviz/lumen/pull/1818))
* Fix prefilter and add tests ([#1797](https://github.com/holoviz/lumen/pull/1797))
* Fix provenance bug ([#1796](https://github.com/holoviz/lumen/pull/1796))
* Prevent error on reconnection ([#1795](https://github.com/holoviz/lumen/pull/1795))
* Fix TableListAgent to list one or more data sources available ([#1872](https://github.com/holoviz/lumen/pull/1872))
* Fix format floats ([#1851](https://github.com/holoviz/lumen/pull/1851))
* Fix copy icon ([#1868](https://github.com/holoviz/lumen/pull/1868))
* Prevent duplicate system prompts ([#1860](https://github.com/holoviz/lumen/pull/1860))
* Handle missing user messages in plan task history rendering ([#1864](https://github.com/holoviz/lumen/pull/1864))

### Documentation

* Panel extensions docs ([#1787](https://github.com/holoviz/lumen/pull/1787))

### Contributors

Many thanks to @philippjfr, @ahuang11, @hoxbro, @maximlt, and @caioaao for their contributions to this release.

**Full changelog:** [v1.1.0 → v1.2.0](https://github.com/holoviz/lumen/compare/v1.1.0...v1.2.0)

## Version 1.1.0

Lumen AI 1.1 builds on the stable foundation of 1.0 with first-class LLM tool support, stronger retrieval-augmented generation, and a broad sweep of usability and reliability improvements.

### LLM Tool Calls & MCP Support

* First-class support for LLM tool calls
    * New `FunctionTool` and `MCPTool` abstractions
* Expanded provider support
    * Dedicated Groq provider
    * Fixed non-streaming Anthropic response handling
    * Fixed annotation support for Google Gemini
    * Graceful multimodal content handling for non-vision providers
    * Cleaner LLM inheritance and provider configuration
    * Default mini model updated to GPT-5.4-mini

### Retrieval-Augmented Generation

* ChromaDB as a vector store backend
    * New `ChromaDBVectorStore` with native embedding model support
    * Shared `BaseOpenAIEmbeddings` base class with normalized whitespace handling
* Smarter metadata handling
    * Per-exploration metadata vector stores for up-to-date context
    * Metadata better surfaced in agent prompts
    * Race condition in `MetadataLookup` indexing resolved
* New `DocumentSummarizerAgent`
    * Document ingestion and summarization as a first-class workflow

### Upload, Export & Editors

* Improved upload experience
    * Configurable file size limits and `FileDropper` options
    * Simplified drag-and-drop flow with better guidance for new users
    * Deduplicated source references for multi-file uploads
* New export formats
    * SQLEditor: JSON and Markdown
    * VegaLiteEditor: WebP, TIFF, and EPS
    * Copy button tooltips now reflect the active editor language

### UX & Interface

* More flexible explorations
    * Explorations can now be created without requiring a data pipeline
    * Direct editing of chat messages via `edit_callback`
    * Label lineage and distinct follow-ups improve conversation traceability
* Refined controls and layout
    * Improved controls abstraction, report UX, and pop-out insertion logic
    * Exploration "prevent and allow" limits for finer administrative control
    * Various prompt improvements for better SQL, revision, and insertion quality

### Data & SQL

* Broader SQL capabilities
    * `BETWEEN` query support via slice syntax in `SQLFilter`
    * Fixed transforms on unselected data and list-typed `source.tables`
* Better table rendering
    * Nested JSON objects flattened correctly, preventing `[object Object]` display
    * Fixed horizontal overflow with long string columns

### Stability

* Pipeline and source fixes
* Broader reliability improvements

### Documentation

* Updated LLM docs and added the AI Catalyst guide
* Documented `template_overrides` with subclassing patterns
* Improved `contributing.md` formatting and fixed a broken README link

### Contributors

Many thanks to @philippjfr and @ahuang11, and a special welcome to the four new contributors making their first commits to the project: @ghostiee-11, @SuMayaBee, @SanketMeghale, and @crypticsaiyan.

**Full changelog:** [v1.0.0 → v1.1.0](https://github.com/holoviz/lumen/compare/v1.0.0...v1.1.0)

## Version 0.10.1
Version 0.10.1 focuses on refining the user experience and improving the robustness of the system. It introduces several bug fixes and minor enhancements to ensure smoother operation and better usability.

### What's Changed
- Add close methods ([#1238](https://github.com/holoviz/lumen/pull/1238))
- Import numpy ([#1239](https://github.com/holoviz/lumen/pull/1239))
- Remove del method ([#1240](https://github.com/holoviz/lumen/pull/1240))
- Handle equality check ([#1242](https://github.com/holoviz/lumen/pull/1242))
- Status badge watch ready ([#1243](https://github.com/holoviz/lumen/pull/1243))
- Add tests for agents used alone ([#1205](https://github.com/holoviz/lumen/pull/1205))
- Implement add_directory method to vector store ([#1221](https://github.com/holoviz/lumen/pull/1221))
- Add status descriptions to badges ([#1246](https://github.com/holoviz/lumen/pull/1246))
- Better initial onboarding experience ([#1249](https://github.com/holoviz/lumen/pull/1249))
- Add back retry ([#1250](https://github.com/holoviz/lumen/pull/1250))
- Add custom source controls ([#1241](https://github.com/holoviz/lumen/pull/1241))
- Minor tweaks to prompts, aesthetics, UX ([#1251](https://github.com/holoviz/lumen/pull/1251))
- More tweaks ([#1252](https://github.com/holoviz/lumen/pull/1252))
- Make retries edit by line ([#1253](https://github.com/holoviz/lumen/pull/1253))
- Optimization tweaks ([#1254](https://github.com/holoviz/lumen/pull/1254))
- Bug fixes and tweaks ([#1256](https://github.com/holoviz/lumen/pull/1256))
- Allow inputting data as URLs ([#1255](https://github.com/holoviz/lumen/pull/1255))

## Version 0.10.0
This release enhances the robustness and functionality of Lumen, focusing on improving the user interface and experience. It introduces new features and improvements to existing functionalities, particularly around vector stores and data handling.

### Features
- Meta context ([#1159](https://github.com/holoviz/lumen/pull/1159))
- Add DbtslLookup and DbtslAgent ([#1160](https://github.com/holoviz/lumen/pull/1160))
- Add global context ([#1181](https://github.com/holoviz/lumen/pull/1181))
- Add situate for contextual RAG ([#1198](https://github.com/holoviz/lumen/pull/1198))

### Enhancements
- Various tweaks ([#1158](https://github.com/holoviz/lumen/pull/1158))
- Fast track column selections ([#1164](https://github.com/holoviz/lumen/pull/1164))
- Share the same vector store ([#1163](https://github.com/holoviz/lumen/pull/1163))
- Add upsert methods to vector stores ([#1170](https://github.com/holoviz/lumen/pull/1170))
- Optimize vector store usage ([#1175](https://github.com/holoviz/lumen/pull/1175))
- Improve vegalite ([#1208](https://github.com/holoviz/lumen/pull/1208))
- Set vector store badge to be ready if no tasks ([#1213](https://github.com/holoviz/lumen/pull/1213))
- Update prompts and models ([#1216](https://github.com/holoviz/lumen/pull/1216))
- Stylize vega-lite ([#1220](https://github.com/holoviz/lumen/pull/1220))

### Bug Fixes
- Fix Dbtsl issues ([#1182](https://github.com/holoviz/lumen/pull/1182))
- Fix status badge ([#1189](https://github.com/holoviz/lumen/pull/1189))
- Fix llama install command in docs ([#1191](https://github.com/holoviz/lumen/pull/1191))
- Fix environment variable name for Anthropic API key ([#1199](https://github.com/holoviz/lumen/pull/1199))
- Handle empty tables ([#1222](https://github.com/holoviz/lumen/pull/1222))
- Missing await coroutine ([#1223](https://github.com/holoviz/lumen/pull/1223))
- Add lock to prevent duckdb corruption ([#1227](https://github.com/holoviz/lumen/pull/1227))
- Fix parquet loading ([#1235](https://github.com/holoviz/lumen/pull/1235))

### Documentation
- Move examples block above ([#1201](https://github.com/holoviz/lumen/pull/1201))

## Version 0.9.0
This release focuses on enhancing the integration and functionality of Lumen with various data sources and embedding models. It introduces new features and improvements to better handle data visualization and user interactions.

### Features
- Add HuggingFaceEmbeddings ([#1093](https://github.com/holoviz/lumen/pull/1093))
- Add WebLLM based LLM provider ([#1153](https://github.com/holoviz/lumen/pull/1153))
- Add BigQuerySource ([#1062](https://github.com/holoviz/lumen/pull/1062))

### Enhancements
- Use model n ctx ([#1083](https://github.com/holoviz/lumen/pull/1083))
- Improve logging ([#1101](https://github.com/holoviz/lumen/pull/1101))
- Improve to work with Sonnet ([#1102](https://github.com/holoviz/lumen/pull/1102))
- Ensure memory callbacks and rx refs are cleaned up ([#1109](https://github.com/holoviz/lumen/pull/1109))
- Refactor ListAgent and add pagination/filters ([#1111](https://github.com/holoviz/lumen/pull/1111))
- Add Source.get_metadata ([#1074](https://github.com/holoviz/lumen/pull/1074))
- Avoid repeatedly looking up and streaming context ([#1043](https://github.com/holoviz/lumen/pull/1043))

### Bug Fixes
- Fix VegaLite retry by moving the log outside ([#1069](https://github.com/holoviz/lumen/pull/1069))
- Fix SQLSelectFrom ([#1117](https://github.com/holoviz/lumen/pull/1117))
- Fix repeat follow up queries, with no errors at all ([#1046](https://github.com/holoviz/lumen/pull/1046))
- Fix missing arg in set_context ([#991](https://github.com/holoviz/lumen/pull/991))
- Fix validation error ([#1050](https://github.com/holoviz/lumen/pull/1050))
- Fix for single table ([#1056](https://github.com/holoviz/lumen/pull/1056))

## Version 0.8.0
This release introduces significant enhancements and new features to Lumen, focusing on improving user interface elements and data handling capabilities. It also includes various bug fixes to enhance stability and performance.

### Features
- Add ability to add files to VectorStore easily ([#973](https://github.com/holoviz/lumen/pull/973))
- Add GraphicWalker view and integrate in AI ExplorerUI ([#1003](https://github.com/holoviz/lumen/pull/1003))
- Add support for more Snowflake authenticators ([#1006](https://github.com/holoviz/lumen/pull/1006))

### Enhancements
- Make ChatInterface configurable at UI level ([#982](https://github.com/holoviz/lumen/pull/982))
- Separate context lookups in Planner ([#983](https://github.com/holoviz/lumen/pull/983))
- Improve VegaLite and fix issue with removing count column ([#1008](https://github.com/holoviz/lumen/pull/1008))
- Ensure extensions are loaded in notebook export ([#1007](https://github.com/holoviz/lumen/pull/1007))
- Do not show error traceback on lumen-ai serve ([#1014](https://github.com/holoviz/lumen/pull/1014))

### Bug Fixes
- Fix suggestions showing ([#974](https://github.com/holoviz/lumen/pull/974))
- Fix async race condition when setting context ([#988](https://github.com/holoviz/lumen/pull/988))
- Fix missing arg in set_context ([#991](https://github.com/holoviz/lumen/pull/991))
- Fix regressions ([#986](https://github.com/holoviz/lumen/pull/986))
- Fix retry ([#992](https://github.com/holoviz/lumen/pull/992))

### Documentation
- Add docs list agent ([#882](https://github.com/holoviz/lumen/pull/882))
- Add docs and fixes for LLM Providers ([#970](https://github.com/holoviz/lumen/pull/970))

## Version 0.7.0
This release marks a significant evolution in Lumen's capabilities by introducing the new lumen.ai subpackage. It transitions Lumen from a declarative dashboarding tool to an extensible agent-based framework, facilitating advanced data interactions and retrieval augmented generation (RAG).

### Features
- This release added the new lumen.ai subpackage to Lumen, marking a shift away from Lumen being solely a declarative dashboarding and monitoring tool. Instead, Lumen is now re-envisioned as an extensible agent-based framework for chatting with data and for retrieval augmented generation (RAG). The declarative nature of Lumen's data model makes it possible for LLMs to easily generate entire data transformation pipelines, visualizations, and many other types of output.

## Version 0.6.0

This release updates Lumen for compatibility with latest Panel and Param releases. Additionally it adds a `DuckDb` source, focuses on a number of improvements for validation and serialization of Lumen specs.

### Features

- Add `DuckDb` source ([#366](https://github.com/holoviz/lumen/pull/366))
- Automatically support for automatically generating filters ([#419](https://github.com/holoviz/lumen/pull/419), [#501](https://github.com/holoviz/lumen/pull/501))
- Add `Sum`, `Count` and `Eval` transforms ([#425](https://github.com/holoviz/lumen/pull/425))
- Add lifecycle callbacks to `Config` ([#429](https://github.com/holoviz/lumen/pull/429), [#441](https://github.com/holoviz/lumen/pull/441))

### Enhancements

- Do not apply range and multi-select filters if it spans entire range ([#420](https://github.com/holoviz/lumen/pull/420))
- Implement support for recursive layout specifications ([#424](https://github.com/holoviz/lumen/pull/424))
- Add a shortcut for downloading a view ([#427](https://github.com/holoviz/lumen/pull/427))
- Add `View.title` option ([#430](https://github.com/holoviz/lumen/pull/430))
- Display loading indicator on `View` ([#435](https://github.com/holoviz/lumen/pull/435))
- Improve `Download` with Index and Filename parameters ([#438](https://github.com/holoviz/lumen/pull/438))
- Allow configuring a `View` data limit ([#443](https://github.com/holoviz/lumen/pull/443))
- Add description to sidebar ([#457](https://github.com/holoviz/lumen/pull/457))
- Sanitize string input in `SQLFilter` ([#463](https://github.com/holoviz/lumen/pull/463))
- Add `config.show_traceback` ([#465](https://github.com/holoviz/lumen/pull/465))
- Allow referencing `Variable` in `Pipeline` table definition ([#478](https://github.com/holoviz/lumen/pull/478))
- Ensure valid key validation is enabled for all components ([#493](https://github.com/holoviz/lumen/pull/493), [#498](https://github.com/holoviz/lumen/pull/498))

### Bug fixes

- Fix how Layouts resolve internal pipelines ([#418](https://github.com/holoviz/lumen/pull/418))
- Correctly sync `WidgetFilter` with url ([#491](https://github.com/holoviz/lumen/pull/491))
- Allow roundtrip of `Layout` `to_spec` and `from_spec` ([#495](https://github.com/holoviz/lumen/pull/495))
- Ensure Lumen yaml paths can be correctly resolved on Windows ([af86b04](https://github.com/holoviz/lumen/commit/af86b044345b5b087689c63fe735f3504ebb6d6b))

### Compatibility

- Compatibility for Panel 1.0 ([#449](https://github.com/holoviz/lumen/pull/449))
- Compatibility with Param 2 ([#475](https://github.com/holoviz/lumen/pull/475))
- Add Python 3.12 and drop Python 3.7 and 3.8 ([#497](https://github.com/holoviz/lumen/pull/497))

## Version 0.5.1

This micro-release fixes a few issues related to the introduction of `Variable` types and `Source` hashing in the Lumen 0.5.0 release.

### Bug fixes

- Dynamically create `Variable` classes to avoid shared state ([#413](https://github.com/holoviz/lumen/pull/413))
- Better handling for shared apply button ([#414](https://github.com/holoviz/lumen/pull/414))
- Add `Source` hash to schema cache file ([#415](https://github.com/holoviz/lumen/pull/415))

### Compatibility

- Update panel pin

### Documentation

- Fixes for penguins and precipitation examples ([#416](https://github.com/holoviz/lumen/pull/416))

## Version 0.5.0

This minor release completely overhauls the internals of Lumen by adding three new concepts:

1. The `Pipeline` which encapsulates any data transformations and filtering applied to a table.
2. `Variable` references which can be used to coordinate controls across a dashboard.
3. `SQLTransform` components which allow pushing compute down to a SQL database.

Additionally this release adds support for validating a specification without actually evaluating it and serializing all Lumen components to a spec.

### Features

- Add ability to expose `View` controls ([#137](https://github.com/holoviz/lumen/pull/137))
- Add `project_lnglat` and `pivot` transforms ([#140](https://github.com/holoviz/lumen/pull/140), [#271](https://github.com/holoviz/lumen/pull/271))
- Add `Altair`, `hvPlotUIView` and `Panel` View ([#141](https://github.com/holoviz/lumen/pull/141), [#250](https://github.com/holoviz/lumen/pull/250), [#287](https://github.com/holoviz/lumen/pull/287))
- Add loading status messages ([#150](https://github.com/holoviz/lumen/pull/150))
- Implement `BinFilter`, `DateFilter` ([#155](https://github.com/holoviz/lumen/pull/155), [#159](https://github.com/holoviz/lumen/pull/159), [#276](https://github.com/holoviz/lumen/pull/276))
- Add ability to sync dashboard state with URL parameters ([#207](https://github.com/holoviz/lumen/pull/207))
- Add `IntakeDremio` source ([#213](https://github.com/holoviz/lumen/pull/213), [#244](https://github.com/holoviz/lumen/pull/244))
- Implement `SQLTransform` ([#222](https://github.com/holoviz/lumen/pull/222), [#232](https://github.com/holoviz/lumen/pull/232), [#384](https://github.com/holoviz/lumen/pull/384))
- Implement `Variable` API ([#242](https://github.com/holoviz/lumen/pull/242), [#275](https://github.com/holoviz/lumen/pull/275), [#347](https://github.com/holoviz/lumen/pull/347), [#394](https://github.com/holoviz/lumen/pull/394))
- Add `throttled` options to `Widget` variable and `Filter` ([#264](https://github.com/holoviz/lumen/pull/264))
- Add `Pipeline` class which encapsulates internal data transformations ([#291](https://github.com/holoviz/lumen/pull/291), [#399](https://github.com/holoviz/lumen/pull/399), [#385](https://github.com/holoviz/lumen/pull/385), [#387](https://github.com/holoviz/lumen/pull/387), [#400](https://github.com/holoviz/lumen/pull/400))
- Implement ability to validate dashboard and pipeline specifications ([#312](https://github.com/holoviz/lumen/pull/312))
- Implement `Component.to_spec` ([#317](https://github.com/holoviz/lumen/pull/317))
- Add precache command ([#340](https://github.com/holoviz/lumen/pull/340))

### Enhancements

- Give AE5 admin users access to all deployments ([#138](https://github.com/holoviz/lumen/pull/138))
- Refactored loading of local modules ([#147](https://github.com/holoviz/lumen/pull/147))
- Do not require auth in dev mode ([#151](https://github.com/holoviz/lumen/pull/151))
- Allow resolving explicit module references in type declarations ([#152](https://github.com/holoviz/lumen/pull/152))
- Refactor internals to allow lazy rendering ([#153](https://github.com/holoviz/lumen/pull/153))
- Add  `config.reloadable` options ([#154](https://github.com/holoviz/lumen/pull/154))
- Warn when no data is available ([#156](https://github.com/holoviz/lumen/pull/156))
- Enable `FlexBox` layout if available ([#157](https://github.com/holoviz/lumen/pull/157))
- Clean up styling of filters ([#158](https://github.com/holoviz/lumen/pull/158))
- Add support for throttling of DateFilter ([#161](https://github.com/holoviz/lumen/pull/161))
- Implement threaded background loading of targets ([#179](https://github.com/holoviz/lumen/pull/179))
- Add support for auth plugins ([#185](https://github.com/holoviz/lumen/pull/185))
- Expose `WidgetFilter` `visible` parameter ([#186](https://github.com/holoviz/lumen/pull/186))
- Only load full schema if needed ([#175](https://github.com/holoviz/lumen/pull/175))
- Expose the disabled property of filters ([#195](https://github.com/holoviz/lumen/pull/195))
- Add control panel for each view ([#196](https://github.com/holoviz/lumen/pull/196))
- Add validation to ensure filters on views actually apply to a view ([#201](https://github.com/holoviz/lumen/pull/201))
- Render only applicable filters when tabbed ([#208](https://github.com/holoviz/lumen/pull/208))
- Hide `Card` header when in `Tabs` ([#210](https://github.com/holoviz/lumen/pull/210))
- Add ability to define `controls` on `Transform` ([#211](https://github.com/holoviz/lumen/pull/211))
- Allow control over whether auth is case-sensitive ([#212](https://github.com/holoviz/lumen/pull/212))
- Add `Transform.apply_to` classmethod ([#220](https://github.com/holoviz/lumen/pull/220))
- Add jobs information to `AE5Source` tables ([#221](https://github.com/holoviz/lumen/pull/221))
- Add support for making `View` data downloadable ([#230](https://github.com/holoviz/lumen/pull/230))
- Allow declaring control options ([#234](https://github.com/holoviz/lumen/pull/234))
- Use `SQLTransforms` to compute SQL Source schemas on database ([#246](https://github.com/holoviz/lumen/pull/246))
- Apply `Filter`s in SQL where applicable ([#255](https://github.com/holoviz/lumen/pull/255))
- Improve and enhance schema caching ([#256](https://github.com/holoviz/lumen/pull/256))
- Allow `Variable`s to reference each other ([#259](https://github.com/holoviz/lumen/pull/259))
- Add `AE5Source` resource allocation table ([#249](https://github.com/holoviz/lumen/pull/249))
- Use `as_materialized` method to materialize `Variable`s ([#257](https://github.com/holoviz/lumen/pull/257))
- Add configurable launcher gallery ([#270](https://github.com/holoviz/lumen/pull/270))
- Group variables by target if appropriate ([#283](https://github.com/holoviz/lumen/pull/283))
- Implement thread locking for caches ([#296](https://github.com/holoviz/lumen/pull/296))
- Fix cache for `dask.dataframe` ([#305](https://github.com/holoviz/lumen/pull/305))
- Notify user of runtime errors ([#322](https://github.com/holoviz/lumen/pull/322))
- Add functionality to manually apply updates ([#330](https://github.com/holoviz/lumen/pull/330))
- Allow for adding widgets as filters programmatically ([#332](https://github.com/holoviz/lumen/pull/332))
- Improve the logic behind importing a module ([#335](https://github.com/holoviz/lumen/pull/335))
- Allow instantiating `Dashboard` with spec ([#363](https://github.com/holoviz/lumen/pull/363))
- Improve support for exporting `View` and `Target` to_spec ([#365](https://github.com/holoviz/lumen/pull/365))
- Allow declaring per query caching behavior at Source level ([#367](https://github.com/holoviz/lumen/pull/367))
- Ensure views are rendered lazily ([#381](https://github.com/holoviz/lumen/pull/381))

### Bug fixes

- Fix `AE5Source` admin user_info (b4d3867)
- Fix `DerivedSource.clear_cache` ([#148](https://github.com/holoviz/lumen/pull/148))
- Fix ncols warning ([#149](https://github.com/holoviz/lumen/pull/149))
- Various fixes for filters ([#162](https://github.com/holoviz/lumen/pull/162))
- Handle booleans in schema ([#164](https://github.com/holoviz/lumen/pull/164))
- Fix filtering logic for multi-range queries ([#160](https://github.com/holoviz/lumen/pull/160))
- Ensure modules persist across sessions ([#165](https://github.com/holoviz/lumen/pull/165))
- Ensure `Tabulator` extension is loaded ([#176](https://github.com/holoviz/lumen/pull/176))
- Fix `IntakeBaseSource.load_schema` option ([#178](https://github.com/holoviz/lumen/pull/178))
- Make robust to warm and autoreload options ([#209](https://github.com/holoviz/lumen/pull/209))
- Deactivate the loading spinner after rerender ([#225](https://github.com/holoviz/lumen/pull/225))
- Fixes and improvements for SQL sources and transforms ([#243](https://github.com/holoviz/lumen/pull/243))
- Add support NULL queries in SQLFilter ([#274](https://github.com/holoviz/lumen/pull/274))
- Fix logic for clearing cache
- Ensure global filters are recreated ([#166](https://github.com/holoviz/lumen/pull/166))
- Fix issue with log out ([#241](https://github.com/holoviz/lumen/pull/241))
- Fixes for schema caching ([#260](https://github.com/holoviz/lumen/pull/260))
- Fix setting of Widget variable default ([#263](https://github.com/holoviz/lumen/pull/263))
- Fixes for reload button layout and functionality ([#298](https://github.com/holoviz/lumen/pull/298))
- Avoid leaking panel config settings ([#349](https://github.com/holoviz/lumen/pull/349))
- Ensure schema is still generated for empty dataframe ([#376](https://github.com/holoviz/lumen/pull/376))
- Ensure error in layout initialization does not break entire dashboard ([#383](https://github.com/holoviz/lumen/pull/383))
- Clean up handling of global and per-session state ([#388](https://github.com/holoviz/lumen/pull/388))
- Small fixes for layout rendering and sorting of enums ([#401](https://github.com/holoviz/lumen/pull/401), [#403](https://github.com/holoviz/lumen/pull/403))

### Backward compatibility

- Rename `Target` to `Layout` ([#393](https://github.com/holoviz/lumen/pull/393))
- Add patch for backwards compatibility with Panel 0.12.6 ([#233](https://github.com/holoviz/lumen/pull/233))

### Documentation

- Initial draft of comparing Lumen to other tools ([#145](https://github.com/holoviz/lumen/pull/145))
- Add Pipeline notebook
- Fix the bikes example ([#204](https://github.com/holoviz/lumen/pull/204))
- Overhaul Lumen documentation and add logo ([#288](https://github.com/holoviz/lumen/pull/288))
- Getting started section ([#323](https://github.com/holoviz/lumen/pull/323))
- initializing the how to section ([#325](https://github.com/holoviz/lumen/pull/325))
- Add reference guide skeleton ([#327](https://github.com/holoviz/lumen/pull/327))
- Adding how to on caching  ([#329](https://github.com/holoviz/lumen/pull/329))
- Docs for data visualization views ([#333](https://github.com/holoviz/lumen/pull/333))
- Add docs for data outtake for downloading data ([#334](https://github.com/holoviz/lumen/pull/334))
- Create core concepts and update tutorial in getting started guide ([#336](https://github.com/holoviz/lumen/pull/336))
- Create how to on custom local components ([#343](https://github.com/holoviz/lumen/pull/343))
- Add reference docs ([#368](https://github.com/holoviz/lumen/pull/368))
- Improve rendering of reference docs ([#409](https://github.com/holoviz/lumen/pull/409))
- Consistently link to reference docs ([#410](https://github.com/holoviz/lumen/pull/410))
- Add earthquake example ([#411](https://github.com/holoviz/lumen/pull/411))

## Version 0.4.1

Minor release:

- Fix filtering of Views
- Add precipitation data files ([#133](https://github.com/holoviz/lumen/pull/133))

## Version 0.4.0

(Relatively) major release:

New features:
- Handle errors while rendering dashboard ([#131](https://github.com/holoviz/lumen/pull/131))
- Defer rendering of dashboard contents until page is rendered ([#123](https://github.com/holoviz/lumen/pull/123))
- Add Melt transform ([#122](https://github.com/holoviz/lumen/pull/122))
- Implement DerivedSource with ability to filter and transform existing sources ([#121](https://github.com/holoviz/lumen/pull/121))
- Add caching to DerivedSource
- Use Datetime pickers ([#119](https://github.com/holoviz/lumen/pull/119))

Bugfixes and minor improvements:

- Clear original source cache on DerivedSource
- Allow providing custom Download labels ([#130](https://github.com/holoviz/lumen/pull/130))
- Fix handling of range filters ([#129](https://github.com/holoviz/lumen/pull/129))
- Unpack panes correctly on Views ([#128](https://github.com/holoviz/lumen/pull/128))
- Fixed dask kwarg on JSONSource ([#127](https://github.com/holoviz/lumen/pull/127))
- Pin python3.8 in conda build env
- Ensure None on widget filter is handled ([#120](https://github.com/holoviz/lumen/pull/120))
- Improve docs ([#112](https://github.com/holoviz/lumen/pull/112))


## Version 0.3.1

Minor release:

- Allow declaring Filter as shared ([#111](https://github.com/holoviz/lumen/pull/111))
- Fix bug inserting Download selector

## Version 0.3.1

Minor release:

- Updated dependencies
- Add Download options to targets ([#110](https://github.com/holoviz/lumen/pull/110))
- Make editable a configurable option ([#109](https://github.com/holoviz/lumen/pull/109))
- Improve docs ([#107](https://github.com/holoviz/lumen/pull/107), [#108](https://github.com/holoviz/lumen/pull/108))
- Gracefully handle missing dask and parquet libraries ([#105](https://github.com/holoviz/lumen/pull/105))

## Version 0.3.0

This is the first "public" release of Lumen.

## Version 0.1.0

This is the first public release of the Lumen project, which provides a framework to build dashboards from a simple yaml specification. It is designed to query information from any source, filter it in various ways and then provide views of that information, which can be anything from a simple indicator to a table or plot.

For now the Lumen project is available only from conda and can be installed with `conda install -c pyviz/label/dev lumen`.
