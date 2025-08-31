import os, glob, json
from pathlib import Path
from typing import Dict, Tuple, List
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.io import from_json

DUMP_DIR = os.environ.get("PLOT_DUMP_DIR", "/srv/plot_dump")

app = Dash(__name__, title="SkyFora Task Dashboard")
server = app.server

try:
    from flask_compress import Compress
    Compress(server)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Caching of figures by filename stem
# -----------------------------------------------------------------------------
_CACHE: Dict[str, Tuple[float, go.Figure]] = {}

def load_figs() -> Dict[str, go.Figure]:
    figs = {}
    for fp in sorted(glob.glob(os.path.join(DUMP_DIR, "*.json"))):
        name = Path(fp).stem
        try:
            mtime = os.path.getmtime(fp)
            cached = _CACHE.get(name)
            if cached and cached[0] == mtime:
                figs[name] = cached[1]
                continue
            with open(fp, "r") as f:
                fig = from_json(f.read())
            _CACHE[name] = (mtime, fig)
            figs[name] = fig
        except Exception as e:
            print(f"⚠️ failed to load {fp}: {e}")
    return figs

# -----------------------------------------------------------------------------
# Grouping: map JSON filename → theme (tab)
# We go by filename prefixes you generate:
#   t1_*  → Wind Energy, t2_* → Solar, t3_* → Insurance, t4_* → Heating,
#   t5_*  → Aviation/Logistics, compare_* → Compare
# Everything else falls into "Misc".
# -----------------------------------------------------------------------------
THEME_LABELS = {
    "t1_": "Wind Energy",
    "t2_": "Solar Energy",
    "t3_": "Insurance Risk",
    "t4_": "District Heating",
    "t5_": "Aviation & Logistics"
}

def theme_for(name: str) -> str:
    for prefix, label in THEME_LABELS.items():
        if name.startswith(prefix):
            return label
    return "Misc"

def index_figs_by_theme(figs: Dict[str, go.Figure]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in sorted(figs.keys()):
        theme = theme_for(name)
        out.setdefault(theme, []).append(name)
    # deterministic order for tabs
    ordered = {}
    for pref in ["Wind Energy","Solar Energy","Insurance Risk","District Heating","Aviation & Logistics"]:
        if pref in out: ordered[pref] = out[pref]
    # append any "Misc" at the end
    for k,v in out.items():
        if k not in ordered: ordered[k] = v
    return ordered

# -----------------------------------------------------------------------------
# Figure cosmetics
# -----------------------------------------------------------------------------
def _tighten_colorbars(fig: go.Figure):
    for tr in fig.data:
        t = (getattr(tr, "type", "") or "").lower()
        if t in ("heatmap", "heatmapgl", "surface", "contour"):
            cb = getattr(tr, "colorbar", None)
            if cb is None or not hasattr(cb, "to_plotly_json"):
                cb = {}
            else:
                cb = dict(cb.to_plotly_json())
            cb.setdefault("x", 1.01)
            cb.setdefault("xpad", 6)
            cb.setdefault("thickness", 12)
            cb.setdefault("len", 0.90)
            cb.setdefault("y", 0.5)
            tr.update(colorbar=cb)

def _anchor_single_panel(fig: go.Figure, name: str):
    # Use your operational BBOX
    x_range = [0.0, 32.0]   # lon_min, lon_max
    y_range = [54.0, 72.0]  # lat_min, lat_max

    has_frames = bool(getattr(fig, "frames", []))
    fig.update_layout(
        uirevision=f"keep:{name}",
        autosize=not has_frames,
        height=650 if has_frames else None,
        margin=dict(l=10, r=10, t=48, b=80 if has_frames else 10),
        dragmode="pan",
        hovermode="closest",
        transition=dict(duration=0),
        xaxis=dict(
            range=x_range,
            constrain="domain",
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            ticks="outside",
            ticklen=4,
            ticksuffix="°E",
            scaleanchor="y",
            scaleratio=1.0
        ),
        yaxis=dict(
            range=y_range,
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            ticks="outside",
            ticklen=4,
            ticksuffix="°N"
        ),
    )
    _tighten_colorbars(fig)

def _anchor_compare(fig: go.Figure, name: str):
    # Same BBOX on both panels
    x_range = [0.0, 31.7]
    y_range = [54.0, 72.0]
    has_frames = bool(getattr(fig, "frames", []))
    fig.update_layout(
        uirevision=f"keep:{name}",
        autosize=False,
        height=680,
        width=None,
        margin=dict(l=10, r=10, t=48, b=80),
        dragmode="pan",
        hovermode="closest",
        transition=dict(duration=0),
        xaxis=dict(domain=[0.00,0.48], range=x_range, constrain="domain", scaleanchor="y", scaleratio=1.0,
                   zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)", ticks="outside", ticklen=4, ticksuffix="°E"),
        yaxis=dict(range=y_range, zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)", ticks="outside", ticklen=4, ticksuffix="°N"),
        xaxis2=dict(domain=[0.52,1.00], range=x_range, constrain="domain", scaleanchor="y2", scaleratio=1.0,
                    zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)", ticks="outside", ticklen=4, ticksuffix="°E"),
        yaxis2=dict(range=y_range, zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)", ticks="outside", ticklen=4, ticksuffix="°N"),
    )
    _tighten_colorbars(fig)

def _anchor_geo(fig: go.Figure, name: str):
    # If two-panel compare (xaxis2 present), anchor both; else single panel
    if "xaxis2" in fig.layout:
        _anchor_compare(fig, name)
    else:
        _anchor_single_panel(fig, name)

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Interval(id="poll", interval=600000, n_intervals=0),  # every 2 min
    html.Div([
        html.H4("SkyFora Task Dashboard", style={"margin":"8px 0 4px 0"}),
        html.Div("Choose a theme and layer", style={"fontSize":"12px","opacity":0.85}),
    ], style={"padding":"6px 10px 2px 10px"}),
    dcc.Store(id="fig-index", storage_type="memory"),
    dcc.Tabs(id="theme-tabs", value=None, className="tabs", parent_style={"marginBottom":"4px"}),
    html.Div([
        html.Div([
            html.Label("Layer", style={"fontSize":"12px","marginRight":"6px"}),
            dcc.Dropdown(id="layer-select", clearable=False, style={"minWidth":"320px"}),
        ], style={"display":"flex","alignItems":"center","gap":"8px","padding":"0 10px 6px 10px"}),
        dcc.Loading(
            dcc.Graph(
                id="graph",
                config={"responsive": True, "scrollZoom": True, "doubleClick": "reset"},
                style={"flex":"1 1 auto","height":"72vh"}
            ),
            type="default"
        ),
    ], style={"display":"flex","flexDirection":"column"})
], style={"height":"100vh","display":"flex","flexDirection":"column"})

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@callback(
    Output("fig-index", "data"),
    Output("theme-tabs", "children"),
    Output("theme-tabs", "value"),
    Input("poll", "n_intervals"),
    prevent_initial_call=False
)
def refresh_index(_):
    figs = load_figs()
    if not figs:
        tabs = [dcc.Tab(label="No figures yet", value="__none__", style={"padding":"4px 10px","fontSize":"12px"})]
        return {"groups":{}, "first": "__none__"}, tabs, "__none__"

    groups = index_figs_by_theme(figs)  # {theme: [name,...]}
    tabs = [dcc.Tab(label=theme, value=theme, style={"padding":"4px 10px","fontSize":"12px"}) for theme in groups.keys()]
    first = next(iter(groups.keys()))
    return {"groups":groups}, tabs, first

@callback(
    Output("layer-select", "options"),
    Output("layer-select", "value"),
    Input("theme-tabs", "value"),
    Input("fig-index", "data"),
)
def update_layer_options(theme, idx):
    if not idx or not theme or "groups" not in idx: return [], None
    names = idx["groups"].get(theme, [])
    if not names: return [], None
    # Use figure titles as labels if available
    figs = load_figs()
    opts = []
    for n in names:
        title = n
        fig = figs.get(n)
        if fig and fig.layout and fig.layout.title and fig.layout.title.text:
            title = fig.layout.title.text
        opts.append({"label": title, "value": n})
    return opts, names[0]

@callback(
    Output("graph", "figure"),
    Input("layer-select", "value"),
    Input("poll", "n_intervals"),
    prevent_initial_call=False
)
def render_layer(name, _):
    figs = load_figs()
    if not figs or not name or name not in figs:
        return go.Figure(layout_title_text="No figures found")
    fig = figs[name]
    _anchor_geo(fig, name)
    return fig

@server.get("/healthz")
def healthz():
    return {"ok": True}