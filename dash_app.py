import os, glob
from pathlib import Path
from typing import Dict, Tuple
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.io import from_json

DUMP_DIR = os.environ.get("PLOT_DUMP_DIR", "/srv/plot_dump")

app = Dash(__name__, title="Plot Gallery")
server = app.server

try:
    from flask_compress import Compress
    Compress(server)
except Exception:
    pass

_CACHE: Dict[str, Tuple[float, go.Figure]] = {}

def load_figs():
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

def _tighten_colorbars(fig: go.Figure):
    for tr in fig.data:
        t = (getattr(tr, "type", "") or "").lower()
        if t in ("heatmap", "heatmapgl", "surface", "contour"):
            # Get the colorbar dict or create a new one
            cb = getattr(tr, "colorbar", None)
            if cb is None:
                cb = {}
            else:
                # Make sure we're working with a dictionary, not a ColorBar object
                if hasattr(cb, "to_plotly_json"):
                    cb = cb.to_plotly_json()
                # Ensure it's a dict
                cb = dict(cb) if isinstance(cb, dict) else {}
            
            # Set default values
            cb.setdefault("x", 1.01)
            cb.setdefault("xpad", 6)
            cb.setdefault("thickness", 12)
            cb.setdefault("len", 0.90)
            cb.setdefault("y", 0.5)
            
            # Update the trace
            tr.update(colorbar=cb)

def _debug_figure_layout(fig, name):
    print(f"\n=== DEBUG: {name} ===")
    print(f"Has frames: {hasattr(fig, 'frames') and bool(fig.frames)}")
    print(f"Has sliders: {hasattr(fig.layout, 'sliders') and bool(fig.layout.sliders)}")
    print(f"Has updatemenus: {hasattr(fig.layout, 'updatemenus') and bool(fig.layout.updatemenus)}")
    if hasattr(fig.layout, 'sliders') and fig.layout.sliders:
        print(f"Slider config: {fig.layout.sliders[0]}")
    print("====================\n")

def _anchor_geo(fig: go.Figure, name: str):
    # Debug: see what we're working with
    _debug_figure_layout(fig, f"Before processing {name}")
    
    # Store original animation controls
    original_sliders = getattr(fig.layout, 'sliders', None)
    original_updatemenus = getattr(fig.layout, 'updatemenus', None)
    original_frames = getattr(fig, 'frames', None)
    
    # Apply your layout changes
    fig.update_layout(
        uirevision=f"keep:{name}",
        autosize=True,
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(constrain="domain", zeroline=False,
                   showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                   ticks="outside", ticklen=4, ticksuffix="°E"),
        yaxis=dict(scaleanchor="x", scaleratio=1.0, zeroline=False,
                   showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                   ticks="outside", ticklen=4, ticksuffix="°N"),
        dragmode="pan", hovermode="closest",
        transition=dict(duration=0)
    )
    
    # Restore animation controls if they existed
    if original_sliders:
        fig.update_layout(sliders=original_sliders)
    if original_updatemenus:
        fig.update_layout(updatemenus=original_updatemenus)
    if original_frames:
        fig.frames = original_frames
    
    _tighten_colorbars(fig)
    
    # Debug: see the result
    _debug_figure_layout(fig, f"After processing {name}")

app.layout = html.Div([
    dcc.Interval(id="poll", interval=120000, n_intervals=0),  # 2 minutes
    dcc.Tabs(
        id="tabs", value=None,
        style={"height":"38px"},
        children=[],
        parent_style={"marginBottom":"4px"},
        className="tabs"
    ),
    dcc.Loading(
        dcc.Graph(
            id="graph",
            config={"responsive": True, "scrollZoom": True, "doubleClick": "reset"},
            style={"flex": "1 1 auto", "height": "100%"}
        ),
        type="default"
    ),
], style={"height": "100vh", "display": "flex", "flexDirection": "column"})

@callback(
    Output("tabs", "children"),
    Output("tabs", "value"),
    Input("poll", "n_intervals")
)
def update_tabs(_):
    figs = load_figs()
    if not figs:
        return [dcc.Tab(label="No figures yet", value="none",
                        style={"padding":"4px 10px", "fontSize":"12px"})], "none"
    names = sorted(figs.keys())
    tabs = [dcc.Tab(label=n, value=n,
                    style={"padding":"4px 10px", "fontSize":"12px"}) for n in names]
    return tabs, names[0]

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
    _anchor_geo(fig, tab_name)  # ensures consistent aspect/spacing
    return fig

@server.get("/healthz")
def healthz():
    return {"ok": True}
