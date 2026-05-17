"""Reusable Plotly chart helpers with crosshair / spike lines."""
import pandas as pd
import plotly.graph_objects as go


SPIKE_COLOR = "#888"


def add_crosshair(fig: go.Figure) -> go.Figure:
    """Enable spike lines on both axes — vertical + horizontal lines that
    follow the cursor and show date + value at any hover position."""
    fig.update_xaxes(
        showspikes=True, spikecolor=SPIKE_COLOR, spikemode="across",
        spikethickness=1, spikedash="dot", spikesnap="cursor",
    )
    fig.update_yaxes(
        showspikes=True, spikecolor=SPIKE_COLOR, spikemode="across",
        spikethickness=1, spikedash="dot", spikesnap="cursor",
    )
    fig.update_layout(hovermode="x", spikedistance=-1)
    return fig


def line_chart(data, title: str = "", y_label: str = "", x_label: str = "",
               height: int = 380) -> go.Figure:
    """Single or multi-series line chart with crosshair."""
    fig = go.Figure()
    if isinstance(data, pd.Series):
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode="lines",
                                 name=data.name or "value", line=dict(width=2)))
    elif isinstance(data, pd.DataFrame):
        for col in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines",
                                     name=str(col), line=dict(width=2)))
    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis_title=x_label or None,
        yaxis_title=y_label or None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return add_crosshair(fig)
