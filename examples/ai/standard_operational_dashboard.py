import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn

import lumen.ai as lmai

from lumen.ai.agents import ChatAgent


# ==========================================================
# 1. PANEL CONFIGURATION
# ==========================================================

pn.extension("tabulator", sizing_mode="stretch_width")

pn.config.raw_css.append("""
    .dashboard-header {
        background: linear-gradient(120deg, #101b35, #24446b);
        color: white;
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 15px;
    }

    .dashboard-header h1 {
        color: white;
        margin: 0;
    }

    .dashboard-header p {
        color: #dbeafe;
        margin-top: 8px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        min-height: 115px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        box-sizing: border-box;
    }

    .section-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-sizing: border-box;
    }

    .anomaly-card {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 5px solid #ea580c;
        border-radius: 10px;
        padding: 18px;
    }

    /* Main two-column layout */
    .dashboard-layout {
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        width: 100%;
        gap: 16px;
        box-sizing: border-box;
    }

    .dashboard-main,
    .assistant-column {
        flex: 1 1 0 !important;
        width: calc(50% - 8px) !important;
        max-width: calc(50% - 8px) !important;
        min-width: 0 !important;
        box-sizing: border-box;
    }

    .dashboard-main,
    .assistant-column,
    .assistant-panel {
        align-self: flex-start !important;
        margin-top: 0 !important;
    }

    .assistant-column,
    .assistant-panel {
        min-width: 0 !important;
        box-sizing: border-box;
    }

    .assistant-panel {
        width: 100%;
        padding: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }

    /* Prevent child content from exceeding its column */
    .dashboard-main > *,
    .assistant-column > *,
    .assistant-panel > * {
        min-width: 0;
        max-width: 100%;
        box-sizing: border-box;
    }

    @media (max-width: 900px) {
        .dashboard-layout {
            flex-direction: column !important;
        }

        .dashboard-main,
        .assistant-column {
            width: 100% !important;
            max-width: 100% !important;
            flex-basis: auto !important;
        }
    }
""")


# ==========================================================
# 2. GENERATE SAMPLE NETWORK DATA
# ==========================================================

np.random.seed(42)

regions = ["North", "South", "East", "West"]

records = []

for hour in range(24):
    for region in regions:

        latency = np.random.normal(45, 8)
        packet_loss = np.random.uniform(0, 2)
        traffic = np.random.normal(500, 80)

        # Introduce one critical network anomaly
        if hour == 18 and region == "North":
            latency = 125
            packet_loss = 8.5
            traffic = 850

        records.append({
            "hour": hour,
            "region": region,
            "latency": round(latency, 2),
            "packet_loss": round(packet_loss, 2),
            "traffic": round(traffic, 2)
        })


data = pd.DataFrame(records)

data["status"] = np.where(
    (data["latency"] > 100) |
    (data["packet_loss"] > 5),
    "Critical",
    "Normal"
)


# ==========================================================
# 3. CALCULATE NETWORK METRICS
# ==========================================================

average_latency = data["latency"].mean()

average_packet_loss = data["packet_loss"].mean()

total_traffic = data["traffic"].sum()

critical_alerts = data[
    data["status"] == "Critical"
].copy()

normal_records = data[
    data["status"] == "Normal"
]

if not critical_alerts.empty:
    anomaly = critical_alerts.iloc[0]
else:
    anomaly = None


# ==========================================================
# 4. DASHBOARD HEADER
# ==========================================================

header = pn.pane.HTML(
    """
    <div class="dashboard-header">
        <h1>🌐 Network Operations Center</h1>
        <p>
            Network monitoring, performance analytics
            and AI-powered anomaly investigation.
        </p>
    </div>
    """,
    sizing_mode="stretch_width"
)


# ==========================================================
# 5. KPI METRIC CARDS
# ==========================================================

def create_metric_card(icon, title, value, unit, description):

    return pn.pane.HTML(
        f"""
        <div class="metric-card">
            <div style="font-size:23px;">{icon}</div>

            <div style="
                font-size:12px;
                font-weight:600;
                color:#64748b;
                margin-top:8px;
            ">
                {title}
            </div>

            <div style="
                font-size:27px;
                font-weight:700;
                color:#1e293b;
                margin-top:7px;
            ">
                {value}
                <span style="
                    font-size:13px;
                    font-weight:500;
                    color:#64748b;
                ">
                    {unit}
                </span>
            </div>

            <div style="
                font-size:11px;
                color:#94a3b8;
                margin-top:5px;
            ">
                {description}
            </div>
        </div>
        """,
        sizing_mode="stretch_width"
    )


latency_card = create_metric_card(
    "⏱️",
    "AVERAGE LATENCY",
    f"{average_latency:.2f}",
    "ms",
    "Average network response time"
)

packet_loss_card = create_metric_card(
    "📉",
    "AVERAGE PACKET LOSS",
    f"{average_packet_loss:.2f}",
    "%",
    "Average packet loss percentage"
)

traffic_card = create_metric_card(
    "📡",
    "TOTAL TRAFFIC",
    f"{total_traffic:,.0f}",
    "Mbps",
    "Combined network traffic"
)

alert_card = create_metric_card(
    "🚨",
    "CRITICAL ALERTS",
    str(len(critical_alerts)),
    "alerts",
    "Detected network anomalies"
)


# Two-column KPI layout
metrics = pn.GridBox(
    latency_card,
    packet_loss_card,
    traffic_card,
    alert_card,
    ncols=2,
    sizing_mode="stretch_width"
)


# ==========================================================
# 6. NETWORK PERFORMANCE CHART
# ==========================================================

latency_chart = data.hvplot.line(
    x="hour",
    y="latency",
    by="region",
    title="Network Latency by Region",
    xlabel="Hour of Day",
    ylabel="Latency (ms)",
    height=340,
    responsive=True,
    legend="top",
    grid=True,
    line_width=2
)

latency_section = pn.Column(
    pn.pane.Markdown("### 📊 Network Performance Trends"),

    pn.pane.HoloViews(
        latency_chart,
        sizing_mode="stretch_width"
    ),

    css_classes=["section-card"],
    sizing_mode="stretch_width"
)


# ==========================================================
# 7. ACTIVE NETWORK ALERTS
# ==========================================================

alert_table = pn.widgets.Tabulator(
    critical_alerts,
    pagination="local",
    page_size=5,
    height=220,
    disabled=True,
    sizing_mode="stretch_width"
)

alerts_section = pn.Column(
    pn.pane.Markdown("### 🚨 Active Network Alerts"),

    pn.pane.Markdown(
        f"""
        **{len(critical_alerts)} critical alert(s)** detected
        in the sample network dataset.
        """
    ),

    alert_table,

    css_classes=["section-card"],
    sizing_mode="stretch_width"
)


# ==========================================================
# 8. ANOMALY INVESTIGATION
# ==========================================================

if anomaly is not None:

    anomaly_details = pn.pane.HTML(
        f"""
        <div class="anomaly-card">

            <h3 style="color:#9a3412;">
                ⚠️ Critical Network Anomaly Detected
            </h3>

            <p><b>Region:</b> {anomaly['region']}</p>

            <p><b>Hour:</b> {int(anomaly['hour'])}:00</p>

            <p><b>Latency:</b> {anomaly['latency']} ms</p>

            <p><b>Packet Loss:</b> {anomaly['packet_loss']}%</p>

            <p><b>Traffic:</b> {anomaly['traffic']} Mbps</p>

            <p><b>Status:</b>
                <span style="color:#dc2626;font-weight:bold;">
                    {anomaly['status']}
                </span>
            </p>

            <hr>

            <b>Recommended Action:</b>

            <p>
                Investigate the {anomaly['region']} region's
                network performance and identify possible
                reasons for increased latency and packet loss.
            </p>

        </div>
        """,
        sizing_mode="stretch_width"
    )

else:

    anomaly_details = pn.pane.Markdown(
        "### ✅ No critical network anomalies detected."
    )


anomaly_section = pn.Column(
    pn.pane.Markdown("### 🔍 Anomaly Investigation"),

    anomaly_details,

    css_classes=["section-card"],
    sizing_mode="stretch_width"
)


# ==========================================================
# 9. LEFT DASHBOARD
# ==========================================================

dashboard = pn.Column(
    header,
    metrics,
    latency_section,
    alerts_section,
    anomaly_section,
    sizing_mode="stretch_width"
)


# ==========================================================
# 10. AI INVESTIGATION ASSISTANT
# ==========================================================

class DashboardExplorerUI(lmai.ExplorerUI):
    """Keep the navigation drawer collapsed when new explorations are added.

    ExplorerUI's default _cleanup_explorations implementation opens the
    navigation drawer when the first exploration is created. This override
    preserves its conversation-synchronization behavior without that
    automatic drawer-opening step.
    """

    async def _cleanup_explorations(self, event):
        if len(event.new) <= len(event.old):
            return

        for i, (old, new) in enumerate(zip(event.old, event.new, strict=False)):
            if old is new:
                continue
            if i == self._last_synced:
                await self._update_conversation()
            break


chat_ui = DashboardExplorerUI(
    data=data,
    default_agents=[ChatAgent],
    title="Network AI Assistant",
    suggestions=[
        "Why did network latency increase?",
        "Which region has the highest packet loss?",
        "Explain the critical network alerts.",
        "Summarize network performance across regions.",
        "Investigate the North region anomaly at hour 18.",
        "Compare network performance across all four regions.",
        "What could cause high latency and packet loss?"
    ]
)

print("ExplorerUI class:", type(chat_ui).__name__)
print("Navigation drawer state:", chat_ui._nav_drawer.open)
print("Navigation drawer width:", chat_ui._nav_drawer.size)

chat_panel = pn.panel(
    chat_ui,
    sizing_mode="stretch_width"
)


# ==========================================================
# 11. RIGHT AI ASSISTANT SECTION
# ==========================================================

assistant_section = pn.Column(
    chat_panel,
    sizing_mode="stretch_width",
    min_width=0,
    css_classes=["assistant-panel"],
)


# ==========================================================
# 12. RESPONSIVE MAIN LAYOUT
# ==========================================================

dashboard_column = pn.Column(
    dashboard,
    sizing_mode="stretch_width",
    min_width=0,
    css_classes=["dashboard-main"],
)

assistant_column = pn.Column(
    assistant_section,
    sizing_mode="stretch_width",
    min_width=0,
    css_classes=["assistant-column"],
)

main_layout = pn.Row(
    dashboard_column,
    assistant_column,
    sizing_mode="stretch_width",
    min_height=800,
    css_classes=["dashboard-layout"],
)


# ==========================================================
# 13. MATERIAL TEMPLATE
# ==========================================================

app = pn.template.MaterialTemplate(
    title="Network Operations Center",
    main_max_width="1600px"
)

app.main.append(main_layout)


# ==========================================================
# 14. SERVE APPLICATION
# ==========================================================

app.servable()
