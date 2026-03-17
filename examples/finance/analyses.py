from __future__ import annotations

import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import param

import lumen.ai as lmai


class RollingReturnAnalysis(lmai.Analysis):
    """
    Plot rolling cumulative return by ticker.
    """

    columns = param.List(default=["timestamp", "ticker", "close"])
    window = param.Integer(default=20, bounds=(2, 252))

    def __call__(self, pipeline, context):
        data = pipeline.data.copy()
        if data.empty:
            return pn.pane.Alert("No rows available for rolling return analysis.", alert_type="warning")

        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values(["ticker", "timestamp"])
        data["return"] = data.groupby("ticker")["close"].pct_change()
        data["rolling_return"] = data.groupby("ticker")["return"].transform(
            lambda series: (1 + series).rolling(int(self.window)).apply(np.prod, raw=True) - 1
        )
        chart = data.dropna(subset=["rolling_return"]).hvplot.line(
            x="timestamp",
            y="rolling_return",
            by="ticker",
            ylabel="Rolling Return",
            title=f"{self.window}-period Rolling Return",
            height=320,
            responsive=True
        )
        return pn.Column(chart)


class VolatilityRegimeAnalysis(lmai.Analysis):
    """
    Plot rolling annualized volatility and regime flags.
    """

    columns = param.List(default=["timestamp", "ticker", "close"])
    vol_window = param.Integer(default=21, bounds=(5, 252))

    def __call__(self, pipeline, context):
        data = pipeline.data.copy()
        if data.empty:
            return pn.pane.Alert("No rows available for volatility analysis.", alert_type="warning")

        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values(["ticker", "timestamp"])
        data["return"] = data.groupby("ticker")["close"].pct_change()
        data["volatility"] = data.groupby("ticker")["return"].transform(
            lambda series: series.rolling(int(self.vol_window)).std() * np.sqrt(252)
        )
        valid = data.dropna(subset=["volatility"]).copy()
        if valid.empty:
            return pn.pane.Alert("Not enough rows to estimate volatility.", alert_type="warning")

        threshold = valid["volatility"].median()
        valid["regime"] = np.where(valid["volatility"] >= threshold, "high_vol", "low_vol")

        vol_line = valid.hvplot.line(
            x="timestamp",
            y="volatility",
            by="ticker",
            title=f"Annualized Volatility ({self.vol_window}-period)",
            height=320,
            responsive=True
        )
        regime_scatter = valid.hvplot.scatter(
            x="timestamp",
            y="volatility",
            by="regime",
            alpha=0.6,
            size=6,
            title="Volatility Regimes",
            height=260,
            responsive=True
        )
        return pn.Column(vol_line, regime_scatter)


class PriceVolumeDistributionAnalysis(lmai.Analysis):
    """
    Plot close-vs-volume relationship by ticker.
    """

    columns = param.List(default=["timestamp", "ticker", "close", "volume"])
    sample = param.Integer(default=2500, bounds=(100, 20000))

    def __call__(self, pipeline, context):
        data = pipeline.data.copy()
        if data.empty:
            return pn.pane.Alert("No rows available for price/volume analysis.", alert_type="warning")

        data = data.dropna(subset=["close", "volume"])
        if len(data) > self.sample:
            data = data.sample(int(self.sample), random_state=42)

        scatter = data.hvplot.scatter(
            x="volume",
            y="close",
            by="ticker",
            alpha=0.5,
            size=7,
            title="Price vs Volume",
            height=320,
            responsive=True
        )
        hist = data.hvplot.hist(
            y="close",
            by="ticker",
            bins=40,
            alpha=0.45,
            title="Close Price Distribution",
            height=280,
            responsive=True
        )
        return pn.Column(scatter, hist)


class OhlcChartAnalysis(lmai.Analysis):
    """
    Render an OHLC chart for a selected ticker.
    """

    columns = param.List(default=["open", "high", "low", "close", "volume"])
    ticker = param.String(default="", doc="Optional ticker filter. Uses first available ticker if unset.")
    max_points = param.Integer(default=240, bounds=(30, 1000))

    def __call__(self, pipeline, context):
        data = pipeline.data.copy()
        if data.empty:
            return pn.pane.Alert("No rows available for OHLC analysis.", alert_type="warning")

        date_col = "timestamp" if "timestamp" in data.columns else "date"
        data["timestamp"] = pd.to_datetime(data[date_col])
        data = data.dropna(subset=["timestamp", "open", "high", "low", "close"])
        if data.empty:
            return pn.pane.Alert("No valid OHLC rows available.", alert_type="warning")

        if "ticker" in data.columns:
            available = sorted(data["ticker"].dropna().astype(str).unique().tolist())
            selected = self.ticker.upper() if self.ticker else available[0]
            if selected not in set(available):
                selected = available[0]
            data = data[data["ticker"].astype(str).str.upper() == selected]
        else:
            selected = "N/A"

        data = data.sort_values("timestamp")
        if len(data) > int(self.max_points):
            data = data.tail(int(self.max_points))

        if data.empty:
            return pn.pane.Alert("No rows available after ticker filtering.", alert_type="warning")

        chart = data.hvplot.ohlc(
            x="timestamp",
            y=["open", "high", "low", "close"],
            title=f"OHLC Chart ({selected})",
            height=420,
            responsive=True,
            tools=["hover"],
            autorange="y",
            grid=True,
            xlabel='',
            xaxis=None
        )
        volume = data.hvplot.bar(
            x="timestamp",
            y="volume",
            height=180,
            responsive=True,
            tools=["hover"],
            grid=True,
            xlabel=''
        )
        return pn.Column(
            pn.pane.HoloViews((chart + volume).cols(1)),
            sizing_mode="stretch_width",
        )
