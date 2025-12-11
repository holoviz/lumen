# :material-view-gallery: Gallery

Example dashboards built with Lumen YAML specifications.

!!! info "About YAML specs"
    Lumen AI runs on Lumen specs under the hood. When you use Lumen AI to create dashboards, it generates these YAML specifications automatically. Dashboards and reports created with Lumen AI are fully reproducible using the generated specs.
    
    **Historically**, users wrote these specs manually before LLMs became capable. Now, Lumen AI can generate them for you through natural language conversation. However, understanding specs is still valuable for:
    
    - **Customizing AI outputs** - Edit and enhance what the AI creates
    - **Programmatic dashboards** - Build dashboards without the AI interface
    - **Version control** - Track changes and reproduce dashboards
    - **Advanced features** - Access features not yet supported by AI
    
    **Choose your approach:**
    
    - Want AI to build it for you? → [Use Lumen AI](../getting_started/using_lumen_ai.md)
    - Want to learn the underlying system? → Continue reading this gallery
    - Want to customize AI-generated dashboards? → Learn from these examples, then edit the YAML

## Featured examples

<div class="grid cards" markdown>

-   :material-penguin: **[Penguins](penguins.md)**

    Palmer Station penguin measurements with linked selections

-   :material-bike: **[London Bike Points](bikes.md)**

    Real-time bike sharing data with interactive maps and time series

-   :material-weather-partly-cloudy: **[Seattle Weather](seattle.md)**

    Historical weather patterns with multi-panel layouts

-   :material-taxi: **[NYC Taxi](nyc_taxi.md)**

    Million-row taxi trip analysis with pickup/dropoff visualizations

-   :material-water: **[Precipitation](precip.md)**

    US precipitation data with geographic visualizations

-   :material-earth: **[Earthquakes](earthquakes.md)**

    Global earthquake data with magnitude and depth analysis

-   :material-wind-turbine: **[Wind Turbines](windturbines.md)**

    Wind turbine performance metrics and geographic distribution

</div>

## How to use these examples

Each example includes:

- Screenshot of the final dashboard
- Complete YAML specification
- Link to download and run locally

**To run an example:**

1. Copy the YAML code
2. Save to a file (e.g., `dashboard.yaml`)
3. Run: `lumen serve dashboard.yaml --show`

**To explore with AI:**

Load the data source and ask Lumen AI to recreate it:

```bash
lumen-ai serve <data-url>
```

Then describe what you want: "Create a dashboard with a scatter plot and histogram" - Lumen AI will generate the YAML spec for you.
