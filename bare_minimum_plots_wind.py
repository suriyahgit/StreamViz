# ─────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────
import os, json, numpy as np, xarray as xr
from pathlib import Path
import plotly.graph_objects as go
import metpy.calc as mpcalc
from metpy.units import units
from plotly.io import to_json
from ecmwf.opendata import Client

# Folder for plots consumed by Dash
PLOT_DUMP_DIR = Path("./plot_dump")
PLOT_DUMP_DIR.mkdir(parents=True, exist_ok=True)

def save_json(fig: go.Figure, name: str):
    path = PLOT_DUMP_DIR / f"{name}.json"
    path.write_text(to_json(fig, pretty=False, validate=False))
    print(f"✅ Saved: {path}")

# ⚙️ Choose a run (UTC)
DATE  = "2025-08-26"        # e.g. '2025-08-26'
TIME  = "18"                # '00'/'06'/'12'/'18'
STEPS = list(range(0, 49, 3))  # 0..48 by 3h

# ROI: Central + Northern Europe (incl. Scandinavia & Latvia)
BBOX = dict(lon_min=5.0, lon_max=32.0, lat_min=47.0, lat_max=71.5)

# Natural Earth paths (download and place these files)
COAST_GEOJSON = "./ne/ne_50m_coastline.json"
ADMIN_GEOJSON = "./ne/ne_50m_admin_0_countries.json"

# Data dir
OUTDIR = "./data/ifs"
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD (optional) & OPEN
# ─────────────────────────────────────────────────────────────
client = Client(source="ecmwf", beta=False)

req_sl = {
    "date":   DATE,
    "time":   TIME,
    "step":   STEPS,
    "type":   "fc",
    "stream": "oper",  # HRES deterministic
    "param":  ["10u","10v","100u","100v","10fg","msl"],
    "target": os.path.join(OUTDIR, f"ifs_sl_{DATE}_{TIME}.grib"),
}
req_slc = {
    "date":   DATE,
    "time":   TIME,
    "step":   STEPS,
    "type":   "fc",
    "stream": "oper",
    "param":  ["100u","100v","msl"],
    "target": os.path.join(OUTDIR, f"ifs_slc_{DATE}_{TIME}.grib"),
}
OUT_PL = os.path.join(OUTDIR, f"ifs_pl_{DATE}_{TIME}.grib2")
req_pl = {
    "date":   DATE, "time": TIME, "step": STEPS,
    "type":   "fc", "stream": "oper",
    "model":  "ifs",
    "grid":   "0.25/0.25",
    "levtype":"pl",
    "levelist":"850/200",
    "param":  ["u","v"],
    "class":  "od", "domain":"g",
    "target": OUT_PL,
}

# If you need to download uncomment:
# client.retrieve(req_sl)
# client.retrieve(req_slc)
# client.retrieve(req_pl)

def open_grib_var(path, short_name, filter_keys=None):
    bk = {"filter_by_keys": {"shortName": short_name}}
    if filter_keys:
        bk["filter_by_keys"].update(filter_keys)
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=bk)
    var = list(ds.data_vars)[0]
    return ds[var]

# --- Single levels
u10  = open_grib_var(req_sl["target"], "10u")
v10  = open_grib_var(req_sl["target"], "10v")
u100 = open_grib_var(req_slc["target"], "100u")
v100 = open_grib_var(req_slc["target"], "100v")
gust = open_grib_var(req_sl["target"], "max_i10fg")
msl  = open_grib_var(req_sl["target"], "msl") / 100.0  # Pa → hPa

# --- Pressure levels
u_pl = open_grib_var(OUT_PL, "u")  # dims: step, isobaricInhPa, latitude, longitude
v_pl = open_grib_var(OUT_PL, "v")
u850 = u_pl.sel(isobaricInhPa=850)
v850 = v_pl.sel(isobaricInhPa=850)
u200 = u_pl.sel(isobaricInhPa=200)
v200 = v_pl.sel(isobaricInhPa=200)

# ─────────────────────────────────────────────────────────────
# 2. SUBSET TO ROI
# ─────────────────────────────────────────────────────────────
def subset_latlon(da, bbox):
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values

    lat_desc = lat_vals[0] > lat_vals[-1]  # ECMWF grids often descending
    lat_slice = slice(bbox["lat_max"], bbox["lat_min"]) if lat_desc else slice(bbox["lat_min"], bbox["lat_max"])
    lon_slice = slice(bbox["lon_min"], bbox["lon_max"])

    return da.sel({lat_name: lat_slice, lon_name: lon_slice})

u10  = subset_latlon(u10,  BBOX)
v10  = subset_latlon(v10,  BBOX)
u100 = subset_latlon(u100, BBOX)
v100 = subset_latlon(v100, BBOX)
gust = subset_latlon(gust, BBOX)
msl  = subset_latlon(msl,  BBOX)
u850 = subset_latlon(u850, BBOX)
v850 = subset_latlon(v850, BBOX)
u200 = subset_latlon(u200, BBOX)
v200 = subset_latlon(v200, BBOX)

lat = u100.latitude.values
lon = u100.longitude.values

# ─────────────────────────────────────────────────────────────
# 3. COASTLINES & BORDERS (static Scattergl overlays)
# ─────────────────────────────────────────────────────────────
def _iter_lines_from_geom(geom):
    t = geom.get("type")
    coords = geom.get("coordinates")
    if t == "LineString":
        yield coords
    elif t == "MultiLineString":
        for line in coords:
            yield line
    elif t == "Polygon":
        if coords:
            yield coords[0]  # exterior ring
    elif t == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]

def _within_bbox(lon, lat, bbox, pad=0.5):
    return (bbox["lon_min"]-pad) <= lon <= (bbox["lon_max"]+pad) and \
           (bbox["lat_min"]-pad) <= lat <= (bbox["lat_max"]+pad)

def geojson_to_scattergl_lines(geojson_path, bbox, name, line_width=1.0, color="rgba(70,70,70,0.9)"):
    traces = []
    if not os.path.exists(geojson_path):
        return traces
    with open(geojson_path, "r") as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        for line in _iter_lines_from_geom(geom):
            xs, ys = [], []
            for lonv, latv in line:
                if _within_bbox(lonv, latv, bbox):
                    xs.append(round(lonv, 3))
                    ys.append(round(latv, 3))
                else:
                    if xs and ys and not (np.isnan(xs[-1]) or np.isnan(ys[-1])):
                        xs.append(np.nan); ys.append(np.nan)
            if len(xs) == 0:
                continue
            traces.append(go.Scattergl(
                x=xs, y=ys, mode="lines",
                line=dict(width=line_width, color=color),
                name=name, hoverinfo="skip", showlegend=False
            ))
    return traces

STATIC_OVERLAYS = []
STATIC_OVERLAYS += geojson_to_scattergl_lines(COAST_GEOJSON, BBOX, "Coastline", line_width=1.1, color="rgba(60,60,60,0.95)")
STATIC_OVERLAYS += geojson_to_scattergl_lines(ADMIN_GEOJSON, BBOX, "Country borders", line_width=0.9, color="rgba(90,90,90,0.8)")

# ─────────────────────────────────────────────────────────────
# 4. HELPERS: times, WebGL arrows, frames w/ slider & trace mapping
# ─────────────────────────────────────────────────────────────
def get_forecast_times(var):
    """Return forecast datetimes (step + time) as datetime64[m]."""
    return (var.step.values + var.time.values).astype("datetime64[m]")

def _quiver_stride(ny, nx, max_arrows=600):
    if max_arrows is None: return 1
    return max(1, int(np.ceil(np.sqrt((ny * nx) / max_arrows))))

def add_vectors_scattergl_arrows(
    fig, u2d, v2d, lat, lon,
    max_arrows=700,     # cap total arrows
    min_speed=None,     # hide calm winds (m/s)
    shaft_len_cell=0.40,# shaft length vs downsampled cell size
    head_len_frac=0.35, # arrow head length as fraction of shaft
    head_angle_deg=25.0,# arrow head opening half-angle
    line_width=1.0,
    name="Wind"
):
    """Add fast WebGL arrows: shaft + two head lines per vector."""
    ny, nx = u2d.shape
    stride = _quiver_stride(ny, nx, max_arrows)

    uu = u2d[::stride, ::stride]
    vv = v2d[::stride, ::stride]
    lats = lat[::stride]
    lons = lon[::stride]
    Lon, Lat = np.meshgrid(lons, lats)

    if min_speed is not None:
        spd = np.hypot(uu, vv)
        mask = spd >= float(min_speed)
    else:
        mask = np.ones_like(uu, dtype=bool)

    x0 = Lon[mask]; y0 = Lat[mask]
    u  = uu[mask];  v  = vv[mask]
    if x0.size == 0:
        return fig

    spd = np.hypot(u, v)
    with np.errstate(divide='ignore', invalid='ignore'):
        ux = np.where(spd > 0, u / spd, 0.0)
        vy = np.where(spd > 0, v / spd, 0.0)

    # grid spacing in degrees
    dx = float(np.nanmean(np.diff(lon))) if lon.size > 1 else 0.25
    dy = float(np.nanmean(np.diff(lat))) if lat.size > 1 else 0.25
    base_len = shaft_len_cell * np.hypot(dx * stride, dy * stride)

    # Shaft end
    x1 = x0 + ux * base_len
    y1 = y0 + vy * base_len

    # Arrowheads: rotate unit vector by ±angle
    ang = np.deg2rad(head_angle_deg)
    cos, sin = np.cos(ang), np.sin(ang)
    # Rotate (ux, vy) by +ang and -ang
    hx1 =  (ux * cos - vy * sin)
    hy1 =  (ux * sin + vy * cos)
    hx2 =  (ux * cos + vy * sin)
    hy2 =  (-ux * sin + vy * cos)
    head_len = head_len_frac * base_len

    # Build polyline arrays with NaN separators: shaft + two head lines
    n = x0.size
    # shaft
    Xs = np.empty(n * 3); Ys = np.empty(n * 3)
    Xs[0::3], Xs[1::3], Xs[2::3] = np.round(x0, 3), np.round(x1, 3), np.nan
    Ys[0::3], Ys[1::3], Ys[2::3] = np.round(y0, 3), np.round(y1, 3), np.nan
    # head 1
    Xh1 = np.empty(n * 3); Yh1 = np.empty(n * 3)
    Xh1[0::3], Xh1[1::3], Xh1[2::3] = np.round(x1, 3), np.round(x1 - hx1 * head_len, 3), np.nan
    Yh1[0::3], Yh1[1::3], Yh1[2::3] = np.round(y1, 3), np.round(y1 - hy1 * head_len, 3), np.nan
    # head 2
    Xh2 = np.empty(n * 3); Yh2 = np.empty(n * 3)
    Xh2[0::3], Xh2[1::3], Xh2[2::3] = np.round(x1, 3), np.round(x1 - hx2 * head_len, 3), np.nan
    Yh2[0::3], Yh2[1::3], Yh2[2::3] = np.round(y1, 3), np.round(y1 - hy2 * head_len, 3), np.nan

    # One Scattergl for all three (reduces trace count): interleave arrays
    X = np.concatenate([Xs, Xh1, Xh2])
    Y = np.concatenate([Ys, Yh1, Yh2])

    fig.add_trace(go.Scattergl(
        x=X, y=Y, mode="lines",
        line=dict(width=line_width),
        name=name, hoverinfo="skip", showlegend=False
    ))
    return fig

def save_slider_plot(frames, title, filename, initial_data):
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=title,
            height=650,
            margin=dict(l=10, r=10, t=40, b=80),  # Increased bottom margin for slider
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            sliders=[{
                "active": 0,
                "steps": [
                    {"method": "animate",
                     "args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                     "label": f.name}
                    for f in frames
                ],
                "transition": {"duration": 0},
                "x": 0, "y": 0,
                "len": 1.0,
                "currentvalue": {"prefix": "Time: ", "font": {"size": 12}, "xanchor": "center"},
                "pad": {"t": 50, "b": 10}
            }],
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶", "method": "animate",
                     "args": [None, {"frame": {"duration": 500}, "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "⏸", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
                "showactive": False,
                "x": 0.1, "y": 0,
                "xanchor": "right", "yanchor": "top",
                "pad": {"r": 10, "t": 10}
            }]
        ),
        frames=frames
    )
    save_json(fig, filename)

# ─────────────────────────────────────────────────────────────
# 5. FRAME BUILDERS (with overlays + traces mapping)
# ─────────────────────────────────────────────────────────────
def build_frames_speed_plus_vectors(
    title, filename, u_da, v_da,
    colorscale="Turbo", z_unit="m/s", zmin=0, zmax=30,
    max_arrows=700, min_speed=None,
    shaft_len_cell=0.40, head_len_frac=0.35, head_angle_deg=25.0,
    line_width=1.0,
    add_mslp=None
):
    times = get_forecast_times(u_da)
    frames = []
    initial_data = None

    lat_arr = u_da.latitude.values
    lon_arr = u_da.longitude.values

    for ti, t in enumerate(times):
        u = u_da.isel(step=ti).metpy.convert_units("m/s").values
        v = v_da.isel(step=ti).metpy.convert_units("m/s").values
        ws = mpcalc.wind_speed(u * units("m/s"), v * units("m/s")).magnitude.astype(np.float32)

        heat = go.Heatmap(
            z=ws, x=lon_arr, y=lat_arr, colorscale=colorscale,
            zmin=zmin, zmax=zmax, colorbar=dict(title=z_unit), showscale=True
        )

        tmp = go.Figure(data=[heat])
        tmp = add_vectors_scattergl_arrows(
            tmp, u, v, lat_arr, lon_arr,
            max_arrows=max_arrows, min_speed=min_speed,
            shaft_len_cell=shaft_len_cell, head_len_frac=head_len_frac,
            head_angle_deg=head_angle_deg, line_width=line_width,
            name="Wind"
        )
        dynamic_traces = list(tmp.data)

        if add_mslp is not None:
            msl_2d = add_mslp.isel(step=ti).values
            vmin = float(np.nanmin(msl_2d)); vmax = float(np.nanmax(msl_2d))
            start = 2 * int(np.floor(vmin / 2)); end = 2 * int(np.ceil(vmax / 2))
            dynamic_traces.append(go.Contour(
                z=msl_2d, x=lon_arr, y=lat_arr,
                contours=dict(start=start, end=end, size=2, coloring="none", showlabels=True),
                line=dict(width=1), showscale=False, name="MSLP"
            ))

        if ti == 0:
            initial_data = STATIC_OVERLAYS + dynamic_traces
            overlays_count = len(STATIC_OVERLAYS)

        # Map frames to dynamic trace slots (after overlays)
        trace_indices = list(range(overlays_count, overlays_count + len(dynamic_traces)))
        time_str = np.datetime_as_string(t, unit="m") + "Z"
        frames.append(go.Frame(data=dynamic_traces, traces=trace_indices, name=time_str))

    save_slider_plot(frames, title, filename, initial_data)

def build_frames_scalar_only(title, filename, da, colorscale, z_unit, zmin, zmax):
    times = get_forecast_times(da)
    frames = []
    initial_data = None
    lat_arr = da.latitude.values
    lon_arr = da.longitude.values

    for ti, t in enumerate(times):
        z = da.isel(step=ti).metpy.convert_units("m/s").values.astype(np.float32)
        heat = go.Heatmap(
            z=z, x=lon_arr, y=lat_arr, colorscale=colorscale,
            zmin=zmin, zmax=zmax, colorbar=dict(title=z_unit), showscale=True
        )
        dynamic_traces = [heat]

        if ti == 0:
            initial_data = STATIC_OVERLAYS + dynamic_traces
            overlays_count = len(STATIC_OVERLAYS)

        trace_indices = list(range(overlays_count, overlays_count + len(dynamic_traces)))
        time_str = np.datetime_as_string(t, unit="m") + "Z"
        frames.append(go.Frame(data=dynamic_traces, traces=trace_indices, name=time_str))

    save_slider_plot(frames, title, filename, initial_data)

def build_frames_shear(u_hi, v_hi, u_lo, v_lo, title, filename, colorscale="Plasma", z_unit="m/s", zmin=0, zmax=20):
    times = get_forecast_times(u_hi)
    frames = []
    initial_data = None
    lat_arr = u_hi.latitude.values
    lon_arr = u_hi.longitude.values

    for ti, t in enumerate(times):
        u100_t = u_hi.isel(step=ti).metpy.convert_units("m/s").values
        v100_t = v_hi.isel(step=ti).metpy.convert_units("m/s").values
        u10_t  = u_lo.isel(step=ti).metpy.convert_units("m/s").values
        v10_t  = v_lo.isel(step=ti).metpy.convert_units("m/s").values
        shear = np.hypot(u100_t - u10_t, v100_t - v10_t).astype(np.float32)

        heat = go.Heatmap(
            z=shear, x=lon_arr, y=lat_arr,
            colorscale=colorscale, zmin=zmin, zmax=zmax,
            colorbar=dict(title=z_unit), showscale=True
        )
        dynamic_traces = [heat]

        if ti == 0:
            initial_data = STATIC_OVERLAYS + dynamic_traces
            overlays_count = len(STATIC_OVERLAYS)

        trace_indices = list(range(overlays_count, overlays_count + len(dynamic_traces)))
        time_str = np.datetime_as_string(t, unit="m") + "Z"
        frames.append(go.Frame(data=dynamic_traces, traces=trace_indices, name=time_str))

    save_slider_plot(frames, title, filename, initial_data)

# ─────────────────────────────────────────────────────────────
# 6. CREATE JSONs (1 per visualisation, all timesteps inside)
# ─────────────────────────────────────────────────────────────

# 100 m wind + MSLP (arrows)
build_frames_speed_plus_vectors(
    title="100 m Wind + MSLP (IFS)",
    filename="100m_wind_msl",
    u_da=u100, v_da=v100,
    colorscale="Turbo", z_unit="m/s", zmin=0, zmax=35,
    max_arrows=600, min_speed=4.0,
    shaft_len_cell=0.45, head_len_frac=0.35, head_angle_deg=25.0,
    line_width=1.0,
    add_mslp=msl
)

# 10 m wind (arrows)
build_frames_speed_plus_vectors(
    title="10 m Wind Speed (IFS)",
    filename="10m_wind",
    u_da=u10, v_da=v10,
    colorscale="Viridis", z_unit="m/s", zmin=0, zmax=25,
    max_arrows=400, min_speed=3.0,
    shaft_len_cell=0.42, head_len_frac=0.35, head_angle_deg=25.0,
    line_width=0.9,
    add_mslp=None
)

# 10 m Gust (heatmap only)
build_frames_scalar_only(
    title="10 m Wind Gust (IFS)",
    filename="10m_gust",
    da=gust,
    colorscale="Turbo", z_unit="m/s", zmin=0, zmax=40
)

# Bulk shear (100m - 10m), heatmap only
build_frames_shear(
    u_hi=u100, v_hi=v100, u_lo=u10, v_lo=v10,
    title="Bulk Wind Shear (100m − 10m)",
    filename="wind_shear_100m_10m",
    colorscale="Plasma", z_unit="m/s", zmin=0, zmax=20
)

# 850 hPa wind (arrows)
build_frames_speed_plus_vectors(
    title="850 hPa Wind Speed (IFS)",
    filename="850hpa_wind",
    u_da=u850, v_da=v850,
    colorscale="Cividis", z_unit="m/s", zmin=0, zmax=40,
    max_arrows=700, min_speed=6.0,
    shaft_len_cell=0.45, head_len_frac=0.35, head_angle_deg=22.5,
    line_width=1.0,
    add_mslp=None
)

# 200 hPa jet (arrows)
build_frames_speed_plus_vectors(
    title="200 hPa Jet (IFS)",
    filename="200hpa_jet",
    u_da=u200, v_da=v200,
    colorscale="Turbo", z_unit="m/s", zmin=0, zmax=80,
    max_arrows=900, min_speed=15.0,
    shaft_len_cell=0.40, head_len_frac=0.33, head_angle_deg=20.0,
    line_width=1.1,
    add_mslp=None
)
