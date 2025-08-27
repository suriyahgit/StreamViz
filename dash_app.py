import os, glob, json, time
from pathlib import Path
from typing import Dict, Tuple
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.io import from_json

DUMP_DIR = os.environ.get("PLOT_DUMP_DIR", "/srv/plot_dump")

app = Dash(__name__, title="Plot Gallery")
server = app.server

# Enable gzip responses
try:
    from flask_compress import Compress
    Compress(server)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# Simple file cache (reload only when mtime changes)
# ─────────────────────────────────────────────────────────────
_CACHE: Dict[str, Tuple[float, go.Figure]] = {}  # {name: (mtime, figure)}

def _safe_from_json(text: str) -> go.Figure:
    # plotly.io.from_json is already optimal; keep validation off in your producer
    return from_json(text)

def _load_one(fp: str):
    name = Path(fp).stem
    try:
        mtime = os.path.getmtime(fp)
        cached = _CACHE.get(name)
        if cached and cached[0] == mtime:
            return name, cached[1]
        with open(fp, "r") as f:
            fig = _safe_from_json(f.read())
        _CACHE[name] = (mtime, fig)
        return name, fig
    except Exception as e:
        print(f"⚠️ failed to load {fp}: {e}")
        return None

def load_figs():
    figs = {}
    for fp in sorted(glob.glob(os.path.join(DUMP_DIR, "*.json"))):
        loaded = _load_one(fp)
        if loaded:
            name, fig = loaded
            figs[name] = fig
    return figs

# ─────────────────────────────────────────────────────────────
# Post-load layout tweaks (no heavy modifications!)
# ─────────────────────────────────────────────────────────────
def _guess_xy_ranges(fig: go.Figure):
    """If ranges are missing, try to infer from first frame/trace with x/y arrays."""
    def read_xy(data_list):
        for tr in data_list:
            x = getattr(tr, "x", None)
            y = getattr(tr, "y", None)
            if x is not None and y is not None and len(x) and len(y):
                try:
                    xmin, xmax = float(min(x)), float(max(x))
                    ymin, ymax = float(min(y)), float(max(y))
                    return [xmin, xmax], [ymin, ymax]
                except Exception:
                    continue
        return None, None

    xr, yr = read_xy(fig.data)
    if xr is None or yr is None:
        # Fallback: first frame
        frames = getattr(fig, "frames", []) or []
        if frames:
            xr, yr = read_xy(frames[0].data)
    return xr, yr

def tune_layout_for_geo(fig: go.Figure, name: str) -> go.Figure:
    # Preserve user state per tab
    fig.update_layout(uirevision=f"keep:{name}", autosize=True)

    # Axis aspect (no horizontal elongation): y anchored to x, 1:1 in degrees
    fig.update_layout(
        xaxis=dict(
            constrain="domain",
            zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.1)",
            ticks="outside", ticklen=4, ticksuffix="°E"
        ),
        yaxis=dict(
            constrain="domain",
            scaleanchor="x", scaleratio=1.0,
            zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.1)",
            ticks="outside", ticklen=4, ticksuffix="°N"
        ),
        margin=dict(l=10, r=10, t=48, b=10),
        dragmode="pan",
        hovermode="closest",
        transition=dict(duration=0)
    )

    # If producer didn't set ranges, infer from data to remove top/bottom whitespace
    xr, yr = _guess_xy_ranges(fig)
    if xr and yr:
        fig.update_xaxes(range=xr)
        fig.update_yaxes(range=yr)

    # Don’t re-touch frames; they’re already optimized WebGL from producer
    return fig

# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Interval(id="poll", interval=60000, n_intervals=0),   # poll every 60s
    dcc.Tabs(id="tabs", value=None),
    dcc.Loading(
        dcc.Graph(
            id="graph",
            config={
                "responsive": True,
                "scrollZoom": True,
                "doubleClick": "reset"
            },
            style={"flex": "1 1 auto", "height": "100%"}
        ), type="default"
    ),
], style={"height": "100vh", "display": "flex", "flexDirection": "column"})

# ─────────────────────────────────────────────────────────────
# Tabs = one per figure file (1 fig = all timesteps)
# ─────────────────────────────────────────────────────────────
@callback(
    Output("tabs", "children"),
    Output("tabs", "value"),
    Input("poll", "n_intervals")
)
def update_tabs(_):
    figs = load_figs()
    if not figs:
        return [dcc.Tab(label="No figures yet", value="none")], "none"
    names = sorted(figs.keys())
    tabs = [dcc.Tab(label=name, value=name) for name in names]
    return tabs, names[0]

# ─────────────────────────────────────────────────────────────
# Render the selected tab, apply geo tuning
# ─────────────────────────────────────────────────────────────
@callback(
    Output("graph", "figure"),
    Input("tabs", "value"),
    Input("poll", "n_intervals")
)
def render_tab(tab_name, _):
    figs = load_figs()
    if not figs or tab_name not in figs:
        return go.Figure(layout_title_text="No figures found")
    fig = figs[tab_name]
    fig = tune_layout_for_geo(fig, tab_name)
    return fig

# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────
@server.get("/healthz")
def healthz():
    return {"ok": True}
