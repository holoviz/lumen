# Standard Operational Dashboard

This example demonstrates how to build a network operations dashboard using Lumen, Panel, and HoloViews.

The dashboard provides an operational view of network performance and integrates Lumen AI to help operators investigate unusual patterns in the underlying data.

## Overview

The dashboard includes:

- **Network Metrics:** Displays average latency, average packet loss, total traffic, and critical alerts.
- **Performance Trends:** Visualizes network latency across four regions: North, South, East, and West.
- **Active Network Alerts:** Displays records that exceed the configured network thresholds.
- **Anomaly Investigation:** Highlights critical network events and provides a recommended investigation action.
- **AI Investigation Assistant:** Integrates Lumen's `ExplorerUI` so operators can explore the dataset and ask questions about network performance.

## Running the Example

Install the project dependencies and run:

```bash
panel serve examples/ai/standard_operational_dashboard.py --show
```

The dashboard opens in your browser.

## AI Configuration

The AI assistant requires a configured LLM provider.

If no provider is configured, the dashboard and its monitoring components remain available, but AI chat functionality is disabled.

Configure an available provider through Lumen's AI settings before using the assistant.

## Sample Anomaly

The example dataset contains an intentionally introduced critical anomaly:

- Region: North
- Hour: 18:00
- Latency: 125 ms
- Packet Loss: 8.5%
- Traffic: 850 Mbps

Operators can use the AI assistant to investigate this event and explore the surrounding network data.

## Implementation

The example uses:

- Panel for dashboard components and layout.
- Pandas and NumPy for sample data generation and processing.
- HoloViews and hvPlot for network performance visualization.
- Lumen AI `ExplorerUI` for interactive data exploration.
- Panel's Material template for the application layout.
