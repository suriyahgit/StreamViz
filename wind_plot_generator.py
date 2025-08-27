# ─────────────────────────────────────────────────────────────
# 0. SETUP  (SVG-only version: no WebGL)
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
DATE  = "2025-08-26"
TIME  = "18"
STEPS = list(range(0, 49, 3))  # 0..48 by 3h

# ROI
BBOX = dict(lon_min=5.0, lon_max=32.0, lat_min=47.0, lat_max=71.5)

# Natural Earth paths
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
    "date":   DATE, "time": TIME, "step": STEPS,
    "type":   "fc",  "stream": "oper",
    "param":  ["10u","10v","100u","100v","10fg","msl"],
    "target": os.path.join(OUTDIR, f"ifs_sl_{DATE}_{TIME}.grib"),
}
req_slc = {
    "date":   DATE, "time": TIME, "step": STEPS,
    "type":   "fc",  "stream": "oper",
    "param":  ["100u","100v","msl"],
    "target": os.path.join(OUTDIR, f"ifs_slc_{DATE}_{TIME}.grib"),
}
OUT_PL = os.path.join(OUTDIR, f"ifs_pl_{DATE}_{TIME}.grib2")
req_pl = {
    "date": DATE, "time": TIME, "step": STEPS,
    "type": "fc", "stream": "oper",
    "model": "ifs", "grid": "0.25/0.25",
    "levtype":"pl", "levelist":"850/200",
    "param": ["u","v"],
    "class": "od", "domain":"g",
    "target": OUT_PL,
}

# If you need to download uncomment:
# client.retrieve(req_sl); client.retrieve(req_slc); client.retrieve(req_pl)

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
u_pl = open_grib_var(OUT_PL, "u")
v_pl = open_grib_var(OUT_PL, "v")
u850 = u_pl.sel(isobaricInhPa=850); v850 = v_pl.sel(isobaricInhPa=850)
u200 = u_pl.sel(isobaricInhPa=200); v200 = v_pl.sel(isobaricInhPa=200)

# ─────────────────────────────────────────────────────────────
# 2. SUBSET TO ROI + ensure monotonic axes for Plotly
# ─────────────────────────────────────────────────────────────
def subset_latlon(da, bbox):
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lat_vals = da[lat_name].values
    lat_desc = lat_vals[0] > lat_vals[-1]  # ECMWF grids often descending
    lat_slice = slice(bbox["lat_max"], bbox["lat_min"]) if lat_desc else slice(bbox["lat_min"], bbox["lat_max"])
    lon_slice = slice(bbox["lon_min"], bbox["lon_max"])
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})

u10  = subset_latlon(u10,  BBOX); v10  = subset_latlon(v10,  BBOX)
u100 = subset_latlon(u100, BBOX); v100 = subset_latlon(v100, BBOX)
gust = subset_latlon(gust, BBOX)
msl  = subset_latlon(msl,  BBOX)
u850 = subset_latlon(u850, BBOX); v850 = subset_latlon(v850, BBOX)
u200 = subset_latlon(u200, BBOX); v200 = subset_latlon(v200, BBOX)

lat_raw = u100.latitude.values
lon_raw = u100.longitude.values

def ensure_xy_monotonic(z2d, lons, lats):
    """Return (z_ascLat, lons, lats_asc). Flip rows if latitude is descending."""
    lons_c = np.array(lons); lats_c = np.array(lats); z2d_c = np.array(z2d)
    if lats_c.size >= 2 and lats_c[0] > lats_c[-1]:
        lats_c = lats_c[::-1]
        z2d_c  = z2d_c[::-1, :]
    return np.ascontiguousarray(z2d_c), np.ascontiguousarray(lons_c), np.ascontiguousarray(lats_c)

# ─────────────────────────────────────────────────────────────
# 3. STATIC OVERLAYS (SVG Scatter)
# ─────────────────────────────────────────────────────────────
# ── REPLACE your geojson_to_scatter_lines + helpers with this version ──


def _iter_lines_from_geom(geom):
    t = geom.get("type"); coords = geom.get("coordinates")
    if t == "LineString":
        yield coords
    elif t == "MultiLineString":
        for line in coords: yield line
    elif t == "Polygon":
        if coords: yield coords[0]          # exterior only
    elif t == "MultiPolygon":
        for poly in coords:
            if poly: yield poly[0]

def _within_bbox(lonv, latv, bbox, pad=0.5):
    return (bbox["lon_min"]-pad) <= lonv <= (bbox["lon_max"]+pad) and \
           (bbox["lat_min"]-pad) <= latv <= (bbox["lat_max"]+pad)

def geojson_to_single_polyline(
    geojson_path, bbox, name,
    line_width=1.0, color="rgba(70,70,70,0.9)",
    decimate_every=2  # keep every N-th coord to speed up rendering (2-4 is fine)
):
    """
    Build ONE go.Scatter with NaN breaks between all segments.
    This prevents Plotly from drawing spurious lines between separate pieces.
    """
    if not os.path.exists(geojson_path):
        return None

    X, Y = [], []
    with open(geojson_path, "r") as f:
        gj = json.load(f)

    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        for line in _iter_lines_from_geom(geom):
            xs, ys = [], []
            # optional decimation for speed
            for i, (lonv, latv) in enumerate(line):
                if decimate_every > 1 and (i % decimate_every) != 0:
                    continue
                if _within_bbox(lonv, latv, bbox):
                    xs.append(round(lonv, 3)); ys.append(round(latv, 3))
                else:
                    # close current subpath if we were inside
                    if xs and ys:
                        xs.append(np.nan); ys.append(np.nan)
            # flush this line
            if xs and ys:
                # ensure a break at the end of every line
                if not (np.isnan(xs[-1]) or np.isnan(ys[-1])):
                    xs.append(np.nan); ys.append(np.nan)
                X.extend(xs); Y.extend(ys)

    if not X:
        return None

    return go.Scatter(
        x=np.array(X), y=np.array(Y),
        mode="lines",
        line=dict(width=line_width, color=color, shape="linear"),
        name=name, hoverinfo="skip", showlegend=False,
        connectgaps=False,  # ← IMPORTANT: never connect across NaNs
        legendgroup="static"
    )


# ── REPLACE your STATIC_OVERLAYS construction ──
STATIC_OVERLAYS = []
coast_trace = geojson_to_single_polyline(COAST_GEOJSON, BBOX, "Coastline", line_width=1.1, color="rgba(60,60,60,0.95)")
borders_trace = geojson_to_single_polyline(ADMIN_GEOJSON, BBOX, "Country borders", line_width=0.9, color="rgba(90,90,90,0.8)")
if coast_trace:   STATIC_OVERLAYS.append(coast_trace)
if borders_trace: STATIC_OVERLAYS.append(borders_trace)


# ─────────────────────────────────────────────────────────────
# 4. SVG ARROWS (quiver) — no WebGL
# ─────────────────────────────────────────────────────────────
def _quiver_stride(ny, nx, max_arrows=600):
    if max_arrows is None: return 1
    return max(1, int(np.ceil(np.sqrt((ny * nx) / max_arrows))))

def add_vectors_svg_arrows(
    fig, u2d, v2d, lat, lon,
    max_arrows=700,
    min_speed=None,
    shaft_len_cell=0.40, head_len_frac=0.35, head_angle_deg=25.0,
    line_width=1.0, name="Wind"
):
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

    dx = float(np.nanmean(np.diff(lon))) if lon.size > 1 else 0.25
    dy = float(np.nanmean(np.diff(lat))) if lat.size > 1 else 0.25
    base_len = shaft_len_cell * np.hypot(dx * stride, dy * stride)

    x1 = x0 + ux * base_len
    y1 = y0 + vy * base_len

    ang = np.deg2rad(head_len_frac * 0 + head_angle_deg)
    cos, sin = np.cos(ang), np.sin(ang)
    hx1 =  (ux * cos - vy * sin); hy1 =  (ux * sin + vy * cos)
    hx2 =  (ux * cos + vy * sin); hy2 =  (-ux * sin + vy * cos)
    head_len = head_len_frac * base_len

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

    X = np.concatenate([Xs, Xh1, Xh2])
    Y = np.concatenate([Ys, Yh1, Yh2])

    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="lines",
        line=dict(width=line_width, shape="linear"),
        name=name, hoverinfo="skip", showlegend=False,
        connectgaps=False
    ))
    return fig

# ─────────────────────────────────────────────────────────────
# 5. FIGURE/FRAME BUILDERS (full data replace, SVG-only)
# ─────────────────────────────────────────────────────────────
def get_forecast_times(var):
    return (var.step.values + var.time.values).astype("datetime64[m]")

def save_slider_plot(frames, title, filename, initial_data):
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title=title,
            height=650,
            margin=dict(l=10, r=10, t=40, b=80),
            xaxis_title="Longitude", yaxis_title="Latitude",
            sliders=[{
                "active": 0,
                "steps": [
                    {"method": "animate",
                     "args": [[f.name],
                              {"frame": {"duration": 0, "redraw": True},
                               "transition": {"duration": 0},
                               "mode": "immediate"}],
                     "label": f.name}
                    for f in frames
                ],
                "transition": {"duration": 0},
                "x": 0, "y": 0, "len": 1.0,
                "currentvalue": {"prefix": "Time: ", "font": {"size": 12}, "xanchor": "center"},
                "pad": {"t": 50, "b": 10}
            }],
            updatemenus=[],            # manual only
            uirevision="keep:svg"      # preserve zoom/pan between frames
        ),
        frames=frames
    )
    save_json(fig, filename)

def _heatmap_from_z(z, x, y, colorscale, zmin, zmax, ztitle):
    return go.Heatmap(
        z=np.ascontiguousarray(z),
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        zsmooth=False,
        hoverinfo="skip",
        colorbar=dict(title=ztitle),
        showscale=True
    )

def _contours_from_z(z, x, y, name="MSLP"):
    vmin = float(np.nanmin(z)); vmax = float(np.nanmax(z))
    start = 2 * int(np.floor(vmin / 2)); end = 2 * int(np.ceil(vmax / 2))
    return go.Contour(
        z=np.ascontiguousarray(z),
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        contours=dict(start=start, end=end, size=2, coloring="none", showlabels=True),
        line=dict(width=1), showscale=False, name=name, connectgaps=False
    )

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

    # t0
    u0 = u_da.isel(step=0).metpy.convert_units("m/s").values
    v0 = v_da.isel(step=0).metpy.convert_units("m/s").values
    ws0 = mpcalc.wind_speed(u0 * units("m/s"), v0 * units("m/s")).magnitude.astype(np.float32)

    ws0_m, lon_m, lat_m = ensure_xy_monotonic(ws0, lon_raw, lat_raw)
    u0_m,  _,     _     = ensure_xy_monotonic(u0,  lon_raw, lat_raw)
    v0_m,  _,     _     = ensure_xy_monotonic(v0,  lon_raw, lat_raw)

    heat0 = _heatmap_from_z(ws0_m, lon_m, lat_m, colorscale, zmin, zmax, z_unit)
    tmp0 = go.Figure(data=[heat0])
    tmp0 = add_vectors_svg_arrows(
        tmp0, u0_m, v0_m, lat_m, lon_m,
        max_arrows=max_arrows, min_speed=min_speed,
        shaft_len_cell=shaft_len_cell, head_len_frac=head_len_frac,
        head_angle_deg=head_angle_deg, line_width=line_width, name="Wind"
    )
    dynamic0 = list(tmp0.data)

    if add_mslp is not None:
        m0 = add_mslp.isel(step=0).values.astype(np.float32)
        m0_m, _, _ = ensure_xy_monotonic(m0, lon_raw, lat_raw)
        dynamic0.append(_contours_from_z(m0_m, lon_m, lat_m, name="MSLP"))

    initial_data = [dynamic0[0]] + STATIC_OVERLAYS + dynamic0[1:]


    # frames
    for ti, t in enumerate(times):
        u = u_da.isel(step=ti).metpy.convert_units("m/s").values
        v = v_da.isel(step=ti).metpy.convert_units("m/s").values
        ws = mpcalc.wind_speed(u * units("m/s"), v * units("m/s")).magnitude.astype(np.float32)

        ws_m, lon_m, lat_m = ensure_xy_monotonic(ws, lon_raw, lat_raw)
        u_m,  _,     _     = ensure_xy_monotonic(u,  lon_raw, lat_raw)
        v_m,  _,     _     = ensure_xy_monotonic(v,  lon_raw, lat_raw)

        heat = _heatmap_from_z(ws_m, lon_m, lat_m, colorscale, zmin, zmax, z_unit)
        tmp = go.Figure(data=[heat])
        tmp = add_vectors_svg_arrows(
            tmp, u_m, v_m, lat_m, lon_m,
            max_arrows=max_arrows, min_speed=min_speed,
            shaft_len_cell=shaft_len_cell, head_len_frac=head_len_frac,
            head_angle_deg=head_angle_deg, line_width=line_width, name="Wind"
        )
        dynamic_traces = list(tmp.data)

        if add_mslp is not None:
            m2 = add_mslp.isel(step=ti).values.astype(np.float32)
            m2_m, _, _ = ensure_xy_monotonic(m2, lon_raw, lat_raw)
            dynamic_traces.append(_contours_from_z(m2_m, lon_m, lat_m, name="MSLP"))

        time_str = np.datetime_as_string(t, unit="m") + "Z"
        frame_data = [dynamic_traces[0]] + STATIC_OVERLAYS + dynamic_traces[1:]
        frames.append(go.Frame(data=frame_data, name=time_str))

    save_slider_plot(frames, title, filename, initial_data)

def build_frames_scalar_only(title, filename, da, colorscale, z_unit, zmin, zmax):
    times = get_forecast_times(da)
    frames = []

    z0 = da.isel(step=0).metpy.convert_units("m/s").values.astype(np.float32)
    z0_m, lon_m, lat_m = ensure_xy_monotonic(z0, lon_raw, lat_raw)
    heat0 = _heatmap_from_z(z0_m, lon_m, lat_m, colorscale, zmin, zmax, z_unit)
    initial_data = [heat0] + STATIC_OVERLAYS

    for ti, t in enumerate(times):
        z = da.isel(step=ti).metpy.convert_units("m/s").values.astype(np.float32)
        z_m, lon_m, lat_m = ensure_xy_monotonic(z, lon_raw, lat_raw)
        heat = _heatmap_from_z(z_m, lon_m, lat_m, colorscale, zmin, zmax, z_unit)
        time_str = np.datetime_as_string(t, unit="m") + "Z"
        frames.append(go.Frame(data=[heat] + STATIC_OVERLAYS, name=time_str))

    save_slider_plot(frames, title, filename, initial_data)

def build_frames_shear(u_hi, v_hi, u_lo, v_lo, title, filename,
                       colorscale="Plasma", z_unit="m/s", zmin=0, zmax=20):
    times = get_forecast_times(u_hi)
    frames = []

    uH = u_hi.isel(step=0).metpy.convert_units("m/s").values
    vH = v_hi.isel(step=0).metpy.convert_units("m/s").values
    uL = u_lo.isel(step=0).metpy.convert_units("m/s").values
    vL = v_lo.isel(step=0).metpy.convert_units("m/s").values
    shear0 = np.hypot(uH - uL, vH - vL).astype(np.float32)
    shear0_m, lon_m, lat_m = ensure_xy_monotonic(shear0, lon_raw, lat_raw)
    heat0 = _heatmap_from_z(shear0_m, lon_m, lat_m, colorscale, zmin, zmax, z_unit)
    initial_data = [heat0] + STATIC_OVERLAYS

    for ti, t in enumerate(times):
        uH = u_hi.isel(step=ti).metpy.convert_units("m/s").values
        vH = v_hi.isel(step=ti).metpy.convert_units("m/s").values
        uL = u_lo.isel(step=ti).metpy.convert_units("m/s").values
        vL = v_lo.isel(step=ti).metpy.convert_units("m/s").values
        shear = np.hypot(uH - uL, vH - vL).astype(np.float32)
        shear_m, lon_m, lat_m = ensure_xy_monotonic(shear, lon_raw, lat_raw)
        heat = _heatmap_from_z(shear_m, lon_m, lat_m, colorscale, zmin, zmax, z_unit)
        time_str = np.datetime_as_string(t, unit="m") + "Z"
        frames.append(go.Frame(data=[heat] + STATIC_OVERLAYS, name=time_str))


    save_slider_plot(frames, title, filename, initial_data)

# ─────────────────────────────────────────────────────────────
# 6. CREATE JSONs
# ─────────────────────────────────────────────────────────────
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
build_frames_scalar_only(
    title="10 m Wind Gust (IFS)",
    filename="10m_gust",
    da=gust,
    colorscale="Turbo", z_unit="m/s", zmin=0, zmax=40
)
build_frames_shear(
    u_hi=u100, v_hi=v100, u_lo=u10, v_lo=v10,
    title="Bulk Wind Shear (100m − 10m)",
    filename="wind_shear_100m_10m",
    colorscale="Plasma", z_unit="m/s", zmin=0, zmax=20
)
build_frames_speed_plus_vectors(
    title="850 hPa Wind (IFS)",
    filename="850hpa_wind",
    u_da=u850, v_da=v850,
    colorscale="Cividis", z_unit="m/s", zmin=0, zmax=40,
    max_arrows=700, min_speed=6.0,
    shaft_len_cell=0.45, head_len_frac=0.35, head_angle_deg=22.5,
    line_width=1.0,
    add_mslp=None
)
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
