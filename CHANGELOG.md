# Releases

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
