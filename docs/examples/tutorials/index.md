# :material-school: Tutorials

Step-by-step guides to build real-world applications with Lumen.

## Available Tutorials

### [Weather Data Explorer](weather_data_ai_explorer.md)
Build a domain-specific data exploration application that analyzes atmospheric soundings and displays Skew-T diagrams. Learn how to create custom agents and tools for specialized data analysis.

**You'll learn:**

- Creating custom agents for domain-specific analysis
- Building specialized visualization tools
- Configuring Lumen for scientific data exploration

### [Census Data AI Explorer](census_data_ai_explorer.md)
Build a custom data source control that integrates U.S. Census Bureau data with Lumen AI. Fetch demographic data through an interactive UI and explore it using natural language queries.

**You'll learn:**

- Creating custom source controls for external APIs
- Building reactive UI with Material-UI components
- Handling async data fetching and loading states
- Registering dynamic data sources with DuckDB
- Best practices for API integration

### [Mesonet Weather Explorer](mesonet_weather_explorer.md)
Build a weather data explorer that fetches daily observations from the Iowa Environmental Mesonet. Learn how to create URL-based source controls with parameter preprocessing.

**You'll learn:**

- Creating URLSourceControls subclasses for REST APIs
- URL template interpolation with typed parameters
- Preprocessing user input before API calls
- Working with date parameters and network selectors

### [Weather API Explorer (OpenAPI)](weather_openapi_explorer.md)
Auto-discover endpoints from the National Weather Service OpenAPI specification — no manual endpoint definitions needed. Point at a spec URL and get a working explorer in ~20 lines.

**You'll learn:**

- Auto-discovering endpoints from an OpenAPI 3.x spec with OpenAPISourceControls
- Filtering endpoints with include_paths and exclude_paths
- How OpenAPI types map to UI widgets
- Combining auto-discovered controls with other source controls

### [Stock Market Explorer](massive_stock_explorer.md)
Wrap the Massive Python SDK to build a stock market data explorer. Pass an SDK client and method names to CodeSourceControls and get auto-generated widgets from method signatures.

**You'll learn:**

- Wrapping Python SDK methods with CodeSourceControls
- How method signatures become UI widgets automatically
- Customizing widgets with param_overrides (dropdowns, bounded sliders)
- Skipping internal SDK parameters with skip_params

### [Build Dashboard with Spec](penguins_dashboard_spec.md)
Create a complete Lumen dashboard in under 15 minutes using YAML specifications. While Lumen AI can generate these specs automatically, understanding them gives you full control.

**You'll learn:**

- Writing YAML specs for dashboards
- Connecting multiple data sources
- Creating interactive visualizations
- Deploying dashboards for sharing

### [SaaS Executive Dashboard](saas_metrics_reports.md)
Turn recurring analytics requests into one-click reproducible reports. Build an executive dashboard that combines SQL queries, custom visualizations, and AI-generated insights—then export to Jupyter notebooks or deploy as a web app.

**You'll learn:**

- Creating custom Actions for specialized visualizations
- Using SQLQuery for deterministic metrics with AI captions
- Wrapping agents with ActorTask for narrative insights
- Organizing reports with Sections and sharing context between tasks

## More Coming Soon

We're constantly adding new tutorials. Check back regularly or [contribute your own](../../reference/contributing.md)!
