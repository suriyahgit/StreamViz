import os, glob
from pathlib import Path
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

def load_figs():
    figs = {}
    for fp in sorted(glob.glob(os.path.join(DUMP_DIR, "*.json"))):
        name = Path(fp).stem
        try:
            with open(fp, "r") as f:
                figs[name] = from_json(f.read())
        except Exception as e:
            print(f"⚠️ failed to load {fp}: {e}")
    return figs

app.layout = html.Div([
    dcc.Interval(id="poll", interval=3000, n_intervals=0),
    dcc.Tabs(id="tabs", value=None),
    dcc.Loading(dcc.Graph(id="graph", config={"responsive": True}), type="default"),
], style={"height":"100vh", "display":"flex", "flexDirection":"column"})

@callback(
    Output("tabs", "children"),
    Output("tabs", "value"),
    Input("poll", "n_intervals")
)
def update_tabs(_):
    figs = load_figs()
    if not figs:
        return [dcc.Tab(label="No figures yet", value="none")], "none"
    tabs = [dcc.Tab(label=name, value=name) for name in figs]
    first = next(iter(figs))
    return tabs, first

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
    fig.update_layout(uirevision="keep", autosize=True,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig

@server.get("/healthz")
def healthz():
    return {"ok": True}
