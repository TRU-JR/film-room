"""PlotBuilder: modular Plotly charts with selectable trendlines (R10, R11)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .errors import PlotError
from .query import label

# Validated categorical palette (light surface). Slots are used in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#9aa3b2"
POSITION_COLORS = {"QB": "#2a78d6", "RB": "#eb6834", "WR": "#1baf7a", "TE": "#6250d6", "DST": "#008300"}
INK, INK_2, MUTED, GRID = "#16233a", "#4a5568", "#7a8597", "#e3e8ef"
FONT = "IBM Plex Sans, system-ui, sans-serif"

CHART_TYPES = {"scatter": "Scatter", "histogram": "Histogram", "box": "Box and whisker",
               "line": "Line", "bar": "Bar"}
TRENDLINES = {None: "None", "linear": "Linear", "polynomial": "Polynomial (degree 2)",
              "exponential": "Exponential"}
MAX_GROUPS = 8


def apply_theme(fig: go.Figure, height: int = 440, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK, size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title_text="", font=dict(size=12, color=INK_2)),
        showlegend=legend, barcornerradius=4, bargap=0.25,
    )
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="",
                tickfont=dict(color=MUTED, size=12), title_font=dict(color=INK_2, size=13))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def color_map(groups: list) -> dict:
    """Fixed colors: positions keep their own hue; other groupings take slots in order."""
    if all(g in POSITION_COLORS for g in groups):
        return {g: POSITION_COLORS[g] for g in groups}
    return {g: (OTHER if g == "Other" else SERIES[i % len(SERIES)]) for i, g in enumerate(groups)}


class PlotBuilder:
    def __init__(self, chart_type: str = "scatter", x: str = "targets", y: str = "fpts",
                 group_by: str | None = None, trendline: str | None = None):
        if chart_type not in CHART_TYPES:
            raise PlotError(f"Unknown chart type '{chart_type}'.")
        self.chart_type = chart_type
        self.x = x
        self.y = y
        self.group_by = group_by
        self.trendline = trendline
        self.r2: float | None = None

    # ----------------------------------------------------------- trendline
    @staticmethod
    def fit_trendline(x, y, kind: str) -> tuple[np.ndarray, float]:
        """Fitted y values (same order as x) and R² for a linear, polynomial or exponential fit."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if kind == "exponential":
            ok &= y > 0
        if ok.sum() < 3 or np.ptp(x[ok]) == 0:
            raise PlotError("A trendline needs at least three points with different x values.")
        xs, ys = x[ok], y[ok]
        if kind == "linear":
            coef = np.polyfit(xs, ys, 1)
            fitted = np.polyval(coef, x)
        elif kind == "polynomial":
            coef = np.polyfit(xs, ys, 2)
            fitted = np.polyval(coef, x)
        elif kind == "exponential":
            b, log_a = np.polyfit(xs, np.log(ys), 1)
            fitted = np.exp(log_a) * np.exp(b * x)
        else:
            raise PlotError(f"Unknown trendline '{kind}'. Use linear, polynomial or exponential.")
        pred = fitted[ok]
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        return fitted, round(r2, 3)

    # ---------------------------------------------------------------- build
    def _groups(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        if not self.group_by:
            return df.assign(_g="All players"), ["All players"]
        d = df.copy()
        d["_g"] = d[self.group_by].astype(str)
        counts = d["_g"].value_counts()
        if len(counts) > MAX_GROUPS:
            keep = set(counts.index[: MAX_GROUPS - 1])
            d.loc[~d["_g"].isin(keep), "_g"] = "Other"
        order = sorted(d["_g"].unique(), key=lambda g: (g == "Other", g))
        return d, order

    def build(self, df: pd.DataFrame) -> go.Figure:
        needed = {"histogram": [self.x], "box": [self.y]}.get(self.chart_type, [self.x, self.y])
        if self.group_by:
            needed.append(self.group_by)
        for c in needed:
            if c not in df:
                raise PlotError(f"The column '{label(c)}' isn't in this data.")
        if df.empty:
            raise PlotError("No rows match the current filters. Widen the filters to draw a chart.")
        d, groups = self._groups(df)
        colors = color_map(groups)
        fig = go.Figure()
        hover_name = "name" if "name" in d else ("team" if "team" in d else None)
        builder = getattr(self, f"_{self.chart_type}")
        builder(fig, d, groups, colors, hover_name)
        if self.trendline and self.chart_type == "scatter":
            self._add_trendline(fig, d)
        xt = label(self.x) if self.chart_type not in ("box",) else (label(self.group_by) if self.group_by else "")
        yt = "Players" if self.chart_type == "histogram" else label(self.y)
        if self.chart_type == "bar" and self.x in ("name", "team"):
            xt, yt = label(self.y), ""
        fig.update_layout(xaxis_title=xt, yaxis_title=yt)
        height = fig.layout.height or 440
        return apply_theme(fig, height=height, legend=len(groups) > 1 or bool(self.trendline))

    def _scatter(self, fig, d, groups, colors, hover_name):
        for g in groups:
            part = d[d["_g"] == g]
            text = part[hover_name] + " " + part["season"].astype(str) if hover_name and "season" in part else (
                part[hover_name] if hover_name else None)
            fig.add_trace(go.Scatter(
                x=part[self.x], y=part[self.y], mode="markers", name=g, text=text,
                marker=dict(size=9, color=colors[g], opacity=0.85, line=dict(width=1.5, color="#ffffff")),
                hovertemplate=f"<b>%{{text}}</b><br>{label(self.x)}: %{{x}}<br>{label(self.y)}: %{{y}}<extra>{g}</extra>"))

    def _histogram(self, fig, d, groups, colors, hover_name):
        for g in groups:
            part = d[d["_g"] == g]
            fig.add_trace(go.Histogram(x=part[self.x], name=g, marker_color=colors[g],
                                       opacity=0.75 if len(groups) > 1 else 0.95, nbinsx=30,
                                       marker_line=dict(width=1, color="#ffffff"),
                                       hovertemplate=f"{label(self.x)}: %{{x}}<br>Players: %{{y}}<extra>{g}</extra>"))
        fig.update_layout(barmode="overlay")

    def _box(self, fig, d, groups, colors, hover_name):
        for g in groups:
            part = d[d["_g"] == g]
            fig.add_trace(go.Box(y=part[self.y], name=g, marker_color=colors[g], boxpoints="outliers",
                                 line=dict(width=2), fillcolor="rgba(255,255,255,0)",
                                 text=part[hover_name] if hover_name else None,
                                 hovertemplate=f"%{{text}}<br>{label(self.y)}: %{{y}}<extra>{g}</extra>"))

    def _line(self, fig, d, groups, colors, hover_name):
        for g in groups:
            part = d[d["_g"] == g].groupby(self.x, as_index=False)[self.y].mean().sort_values(self.x)
            fig.add_trace(go.Scatter(x=part[self.x], y=part[self.y].round(2), mode="lines+markers", name=g,
                                     line=dict(width=2.5, color=colors[g]),
                                     marker=dict(size=8, color=colors[g], line=dict(width=2, color="#ffffff")),
                                     hovertemplate=f"{label(self.x)}: %{{x}}<br>{label(self.y)}: %{{y}}<extra>{g}</extra>"))
        if self.x == "season":
            fig.update_xaxes(dtick=1)

    def _bar(self, fig, d, groups, colors, hover_name):
        horizontal = self.x in ("name", "team")
        d = d.sort_values(self.y, ascending=False)
        is_pct = self.y.endswith("_share") or self.y == "catch_rate"
        fmt = "%{value:.0%}" if is_pct else "%{value:,.4~g}"
        for g in groups:
            part = d[d["_g"] == g]
            if horizontal:
                fig.add_trace(go.Bar(y=part[self.x], x=part[self.y], name=g, orientation="h",
                                     marker_color=colors[g], texttemplate=fmt, textposition="outside",
                                     textfont=dict(color=INK_2, size=12), cliponaxis=False,
                                     hovertemplate=f"<b>%{{y}}</b><br>{label(self.y)}: %{{x}}<extra>{g}</extra>"))
            else:
                fig.add_trace(go.Bar(x=part[self.x], y=part[self.y], name=g, marker_color=colors[g],
                                     hovertemplate=f"<b>%{{x}}</b><br>{label(self.y)}: %{{y}}<extra>{g}</extra>"))
        if horizontal:
            fig.update_yaxes(categoryorder="array", categoryarray=list(d[self.x]), autorange="reversed")
            fig.update_layout(height=max(320, 34 * len(d) + 90))
        else:
            fig.update_xaxes(categoryorder="array", categoryarray=list(d[self.x]))

    def _add_trendline(self, fig, d):
        fitted, r2 = self.fit_trendline(d[self.x], d[self.y], self.trendline)
        self.r2 = r2
        order = np.argsort(d[self.x].to_numpy(dtype=float))
        name = f"{TRENDLINES[self.trendline].split(' (')[0]} fit, R² = {r2:.2f}"
        fig.add_trace(go.Scatter(x=d[self.x].to_numpy()[order], y=fitted[order], mode="lines", name=name,
                                 line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
