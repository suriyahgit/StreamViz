# ──────────────────────────────────────────────────────────────────────────────
# Thematic Dashboard JSON Generator (MEPS, EPSG:4326)
#   • Tabs: Wind Energy, Solar Energy, Insurance Risk, District Heating, Aviation
#   • Scalars → pre-colorized PNG per frame (tiny JSON)
#   • Vectors → SVG arrows in every frame (decimated & rounded)
#   • Static overlays (NE 50m) drawn once
#   • First 24 steps
#   • Standard Plotly colormaps
#   • Outputs Plotly JSON into ./plot_dump
# ──────────────────────────────────────────────────────────────────────────────
import os, io, json, base64, re
from pathlib import Path
import numpy as np
import xarray as xr
from PIL import Image

import plotly.graph_objects as go
from plotly.io import to_json
from plotly import colors as pcolors
from plotly.express import colors as pxcolors

import pystac_client

# =========================
# CONFIG
# =========================
PLOT_DUMP_DIR = Path("./plot_dump")
PLOT_DUMP_DIR.mkdir(parents=True, exist_ok=True)

NE_COAST = "./ne/ne_50m_coastline.json"
NE_BORD  = "./ne/ne_50m_admin_0_countries.json"

MAX_ARROWS     = 450     # per frame
ARROW_MIN_SPD  = 2.5     # m/s threshold for drawing arrows
ARROW_LEN_DEG  = 0.30    # arrow shaft visual length in degrees
ARROW_HEAD_FR  = 0.33
ARROW_HEAD_DEG = 24.0
ARROW_WIDTH    = 1.0

# Standard colorscale choices per product
CMAP = {
    "wind":   "Viridis",
    "jet":    "Viridis",
    "gust":   "Turbo",
    "precip": "PuBuGn",
    "cloud":  "Greys",
    "alt":    "Cividis",
    "ssrd":   "YlOrRd",
    "strd":   "YlGnBu",
    "shear":  "Plasma",
    "hdh":    "YlOrBr",
    "apparent":"RdBu_r",
    "freeze":"Blues",
}

# =========================
# STAC LOAD (datasets already in EPSG:4326)
# =========================
catalog = pystac_client.Client.open("http://localhost:8081")

# Pressure levels
pl_items = list(catalog.search(collections=["MEPS_DET_PRESSURE_2_5KMS"]).items())
ds_meps_pl = xr.open_zarr(pl_items[0].assets["data"].href).isel(time=0).compute()

# Surface/single levels
sl_items = list(catalog.search(collections=["MEPS_DET_SINGLE_2_5KMS"]).items())
ds_meps_sfc = xr.open_zarr(sl_items[0].assets["data"].href).isel(time=0).compute()

# Keep only first 24 steps
def head24(da): return da.isel(step=slice(0, min(24, da.sizes["step"])))

# Coordinates (rectilinear lon/lat or x/y)
def get_lonlat(ds):
    if "lon" in ds.coords and "lat" in ds.coords:
        return np.asarray(ds["lon"].values), np.asarray(ds["lat"].values)
    if "x" in ds.coords and "y" in ds.coords:
        return np.asarray(ds["x"].values),  np.asarray(ds["y"].values)
    raise ValueError("Dataset has no (lon,lat) or (x,y) coords")

LON, LAT = get_lonlat(ds_meps_sfc)
LON = np.around(LON, 3); LAT = np.around(LAT, 3)

# =========================
# Color utilities
# =========================
def _parse_rgb_str(c):
    if isinstance(c, (tuple, list)):
        r, g, b = c[:3]; return int(r), int(g), int(b)
    c = str(c).strip()
    if c.startswith("#") and len(c) == 7:
        return int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
    m = re.match(r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)", c)
    if m:
        return int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3)))
    return _parse_rgb_str(pcolors.label_rgb([c])[0])

def LUT_from_colorscale(name_or_list, n=256):
    cs = pcolors.get_colorscale(name_or_list) if isinstance(name_or_list, str) else name_or_list
    xs = np.array([stop[0] for stop in cs], dtype=float)
    rgb = np.array([_parse_rgb_str(stop[1]) for stop in cs], dtype=float)
    xi = np.linspace(0.0, 1.0, n)
    r = np.interp(xi, xs, rgb[:,0]); g = np.interp(xi, xs, rgb[:,1]); b = np.interp(xi, xs, rgb[:,2])
    return np.stack([r, g, b], axis=1).astype(np.uint8)

def plotly_colorscale(name_or_list):
    return pcolors.get_colorscale(name_or_list) if isinstance(name_or_list, str) else name_or_list

def LUT_from_qualitative(name="Set3", k=12):
    pal = pxcolors.qualitative.__dict__.get(name, pxcolors.qualitative.Set3)
    lut = np.zeros((256,3), dtype=np.uint8)
    for i, c in enumerate(pal[:k]):
        lut[i] = _parse_rgb_str(c)
    return lut

# =========================
# Overlays (NE 50m) — added ONCE
# =========================
def _iter_lines(geom):
    t, coords = geom.get("type"), geom.get("coordinates")
    if t == "LineString": yield coords
    elif t == "MultiLineString":
        for c in coords: yield c
    elif t == "Polygon":
        if coords: yield coords[0]
    elif t == "MultiPolygon":
        for poly in coords:
            if poly: yield poly[0]

def make_overlay(geojson_path, name, width=1.0, color="rgba(60,60,60,0.9)", dec=2):
    if not os.path.exists(geojson_path): return None
    with open(geojson_path,"r") as f: gj = json.load(f)
    X, Y = [], []
    for feat in gj.get("features", []):
        for line in _iter_lines(feat.get("geometry", {})):
            xs, ys = [], []
            for i, (lon, lat) in enumerate(line):
                if dec > 1 and (i % dec): continue
                xs.append(round(lon, 3)); ys.append(round(lat, 3))
            if xs:
                xs.append(np.nan); ys.append(np.nan)
                X.extend(xs); Y.extend(ys)
    if not X: return None
    return go.Scatter(x=np.array(X), y=np.array(Y), mode="lines",
                      line=dict(width=width, color=color), hoverinfo="skip",
                      showlegend=False, name=name, connectgaps=False)

STATIC_OVERLAYS = []
coast = make_overlay(NE_COAST, "Coastline", 1.0, "rgba(60,60,60,0.95)", 2)
bords = make_overlay(NE_BORD,  "Borders",   0.9, "rgba(90,90,90,0.80)", 2)
if coast: STATIC_OVERLAYS.append(coast)
if bords: STATIC_OVERLAYS.append(bords)

# =========================
# PNG helpers
# =========================
def norm_to_uint8(z, vmin, vmax):
    z = np.asarray(z, np.float32)
    z = (z - vmin) / max(vmax - vmin, 1e-6)
    z = np.clip(z, 0, 1)
    return (z * 255.0 + 0.5).astype(np.uint8)

def colorize_uint8(gray_u8, lut256):
    return lut256[gray_u8]  # (H,W,3)

def png_b64_from_rgb(rgb):
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def image_trace_from_png(b64, lon, lat):
    return go.Image(
        source=b64,
        x0=float(lon[0]), y0=float(lat[0]),
        dx=float(lon[1]-lon[0]) if len(lon)>1 else 0.05,
        dy=float(lat[1]-lat[0]) if len(lat)>1 else 0.05
    )

# =========================
# Arrows (SVG per frame)
# =========================
def quiver_stride(ny, nx, max_arrows):
    return max(1, int(np.ceil(np.sqrt((ny * nx) / max_arrows))))

def arrow_trace(u2d, v2d, lon, lat,
                max_arrows=MAX_ARROWS, min_speed=ARROW_MIN_SPD,
                shaft_len_deg=ARROW_LEN_DEG, head_len_frac=ARROW_HEAD_FR,
                head_angle_deg=ARROW_HEAD_DEG, line_width=ARROW_WIDTH):
    ny, nx = u2d.shape
    stride = quiver_stride(ny, nx, max_arrows)
    uu = u2d[::stride, ::stride]; vv = v2d[::stride, ::stride]
    Lon, Lat = np.meshgrid(lon[::stride], lat[::stride])

    spd = np.hypot(uu, vv)
    mask = spd >= float(min_speed)
    x0, y0, u, v = Lon[mask], Lat[mask], uu[mask], vv[mask]
    if x0.size == 0:
        return go.Scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip")

    spd = np.hypot(u, v) + 1e-12
    ux, vy = u / spd, v / spd

    dx = float(np.nanmean(np.diff(lon))) if lon.size>1 else 0.05
    dy = float(np.nanmean(np.diff(lat))) if lat.size>1 else 0.05
    base_len = shaft_len_deg * np.hypot(dx * stride, dy * stride)

    x1, y1 = x0 + ux * base_len, y0 + vy * base_len

    ang = np.deg2rad(head_angle_deg); cos, sin = np.cos(ang), np.sin(ang)
    hx1 = (ux * cos - vy * sin); hy1 = (ux * sin + vy * cos)
    hx2 = (ux * cos + vy * sin); hy2 = (-ux * sin + vy * cos)
    head_len = head_len_frac * base_len

    n = x0.size
    Xs = np.empty(n * 3); Ys = np.empty(n * 3)
    Xs[0::3], Xs[1::3], Xs[2::3] = x0, x1, np.nan
    Ys[0::3], Ys[1::3], Ys[2::3] = y0, y1, np.nan
    Xh1 = np.empty(n * 3); Yh1 = np.empty(n * 3)
    Xh1[0::3], Xh1[1::3], Xh1[2::3] = x1, x1 - hx1 * head_len, np.nan
    Yh1[0::3], Yh1[1::3], Yh1[2::3] = y1, y1 - hy1 * head_len, np.nan
    Xh2 = np.empty(n * 3); Yh2 = np.empty(n * 3)
    Xh2[0::3], Xh2[1::3], Xh2[2::3] = x1, x1 - hx2 * head_len, np.nan
    Yh2[0::3], Yh2[1::3], Yh2[2::3] = y1, y1 - hy2 * head_len, np.nan

    X = np.around(np.concatenate([Xs, Xh1, Xh2]), 3)
    Y = np.around(np.concatenate([Ys, Yh1, Yh2]), 3)

    return go.Scatter(x=X, y=Y, mode="lines", line=dict(width=line_width),
                      hoverinfo="skip", showlegend=False, name="Wind", connectgaps=False)

# =========================
# Figure builder
# =========================
def save_json(fig: go.Figure, name: str):
    path = PLOT_DUMP_DIR / f"{name}.json"
    path.write_text(to_json(fig, pretty=False, validate=False))
    print(f"✅ Saved: {path}")

def build_png_plus_arrows(
    title, filename,
    scalar_da,                # (step, lat, lon)
    colorscale_name, vmin, vmax, unit,
    u_da=None, v_da=None,     # (step, lat, lon) if vectors desired
    categorical=False         # if True, skip colorbar
):
    lut256 = None if categorical else LUT_from_colorscale(colorscale_name)

    steps = scalar_da.step.values[:min(24, scalar_da.sizes["step"])]
    nsteps = len(steps)

    # t0
    z0 = scalar_da.isel(step=0).values
    if categorical:
        # qualitative LUT for categories [0..K]
        vmax_int = int(np.nanmax(z0)) if np.isfinite(np.nanmax(z0)) else 0
        lut_cat = LUT_from_qualitative("Set3", k=max(1, vmax_int+1))
        z0_i = np.clip(np.asarray(z0, dtype=np.int32), 0, 255)
        rgb0 = lut_cat[z0_i]
    else:
        g0 = norm_to_uint8(z0, vmin, vmax)
        rgb0 = colorize_uint8(g0, lut256)
    img0 = image_trace_from_png(png_b64_from_rgb(rgb0), LON, LAT)

    data0 = [img0]
    if (u_da is not None) and (v_da is not None):
        u0 = u_da.isel(step=0).values; v0 = v_da.isel(step=0).values
        data0.append(arrow_trace(u0, v0, LON, LAT))

    data0 += STATIC_OVERLAYS

    if not categorical:
        cbar = go.Heatmap(
            z=[[vmin, vmax]], x=[0,1], y=[0,1],
            showscale=True, hoverinfo="skip", visible=True,
            colorscale=plotly_colorscale(colorscale_name),
            colorbar=dict(title=unit)
        )
        data0.append(cbar)

    frames = []
    for ti in range(nsteps):
        z = scalar_da.isel(step=ti).values
        if categorical:
            z_i = np.clip(np.asarray(z, dtype=np.int32), 0, 255)
            rgb = lut_cat[z_i]
        else:
            gz = norm_to_uint8(z, vmin, vmax)
            rgb = colorize_uint8(gz, lut256)
        img = image_trace_from_png(png_b64_from_rgb(rgb), LON, LAT)

        if (u_da is not None) and (v_da is not None):
            u = u_da.isel(step=ti).values; v = v_da.isel(step=ti).values
            frames.append(go.Frame(
                data=[img, arrow_trace(u, v, LON, LAT)],
                traces=[0,1],
                name=np.datetime_as_string(steps[ti], unit="m")
            ))
        else:
            frames.append(go.Frame(
                data=[img],
                traces=[0],
                name=np.datetime_as_string(steps[ti], unit="m")
            ))

    fig = go.Figure(
        data=data0,
        layout=go.Layout(
            title=title,
            height=650,
            xaxis_title="Longitude", yaxis_title="Latitude",
            uirevision="keep:svg",
            sliders=[{
                "active": 0,
                "steps": [{
                    "method": "animate",
                    "args": [[f.name], {"frame":{"duration":0,"redraw":True},
                                        "transition":{"duration":0},"mode":"immediate"}],
                    "label": f.name
                } for f in frames],
                "x":0,"y":0,"len":1.0,
                "currentvalue":{"prefix":"Time: ","font":{"size":12}}
            }],
            updatemenus=[]
        ),
        frames=frames
    )
    save_json(fig, filename)

# =========================
# Derived metrics
# =========================
def hub_height_speed(u10, v10, land_frac=None, hub_m=100.0, alpha_land=0.20, alpha_sea=0.11):
    """Simple power law using alpha map from land/sea fraction."""
    spd10 = xr.apply_ufunc(np.hypot, u10, v10)
    if land_frac is None:
        alpha = alpha_land  # conservative on land
    else:
        alpha = land_frac * alpha_land + (1.0 - land_frac) * alpha_sea
    ratio = (hub_m / 10.0) ** alpha
    return spd10 * ratio

def capacity_factor_from_speed(v, v_cut_in=3.5, v_rated=12.0, v_cut_out=25.0):
    """Very simple generic curve → [0..1]."""
    v = xr.where(np.isfinite(v), v, 0.0)
    cf = xr.zeros_like(v)
    cf = xr.where(v < v_cut_in, 0.0, cf)
    mid = (v >= v_cut_in) & (v < v_rated)
    cf = xr.where(mid, ((v - v_cut_in) / (v_rated - v_cut_in)) ** 3, cf)
    cf = xr.where((v >= v_rated) & (v < v_cut_out), 1.0, cf)
    cf = xr.where(v >= v_cut_out, 0.0, cf)
    return cf.clip(0, 1)

def gust_factor(gust, u10, v10):
    spd10 = xr.apply_ufunc(np.hypot, u10, v10)
    return (gust / (spd10 + 1e-6)).clip(0, 3)

def ramp_3h(vstep):  # abs change over 3 hours (MEPS hourly)
    r = vstep.diff("step", n=3)
    return xr.apply_ufunc(np.abs, r).pad(step=(3,0))

def rate_from_integral(int_da):  # J/m² over 1h → W/m²
    return int_da.diff("step") / 3600.0 if int_da.sizes["step"] > 1 else int_da * 0

def precip_rate_pref_components(ds):
    rain = ds.get("rainfall_amount")
    snow = ds.get("snowfall_amount")
    grau = ds.get("graupelfall_amount")
    tp_acc = ds.get("precipitation_amount_acc")
    if any(v is not None for v in [rain, snow, grau]):
        parts = []
        if rain is not None: parts.append(head24(rain).fillna(0))
        if snow is not None: parts.append(head24(snow).fillna(0))
        if grau is not None: parts.append(head24(grau).fillna(0))
        out = xr.zeros_like(parts[0])
        for p in parts: out = out + p
        return out
    elif tp_acc is not None:
        return head24(tp_acc).diff("step").pad(step=(1,0))
    else:
        return None

def freezing_fraction(t2m):
    """Fraction of next 24h with T2M < 0°C → 0..1 (deterministic frequency surrogate)."""
    cold = (head24(t2m) < 273.15).astype("float32")
    return cold.mean("step")

def heating_degree_hours_24h(t2m, base_c=18.0):
    t_c = head24(t2m) - 273.15
    hdh = xr.where(t_c < base_c, base_c - t_c, 0.0).sum("step")
    return hdh  # °C·h

def deep_shear(u_pl, v_pl, p_low, p_high):
    return xr.apply_ufunc(np.hypot,
                          head24(u_pl.sel(pressure=p_low) - u_pl.sel(pressure=p_high)),
                          head24(v_pl.sel(pressure=p_low) - v_pl.sel(pressure=p_high)))

def llj_max(u_pl, v_pl, levels=(925.0, 900.0, 875.0, 850.0, 800.0, 775.0, 750.0, 700.0)):
    levs = [p for p in levels if float(p) in u_pl.pressure.values]
    spds = xr.concat([xr.apply_ufunc(np.hypot, head24(u_pl.sel(pressure=p)), head24(v_pl.sel(pressure=p))) for p in levs],
                     dim="plev_llj").assign_coords(plev_llj=levs)
    return spds.max("plev_llj")

def ceiling_risk_fraction(cloud_base_m, thresh_m=300.0):
    cb = head24(cloud_base_m)
    return (cb < thresh_m).astype("float32").mean("step")

# =========================
# Build products per Theme
# =========================

# --- common handles
u10 = head24(ds_meps_sfc["x_wind_10m"]) if "x_wind_10m" in ds_meps_sfc else None
v10 = head24(ds_meps_sfc["y_wind_10m"]) if "y_wind_10m" in ds_meps_sfc else None
gust = head24(ds_meps_sfc["wind_speed_of_gust"]) if "wind_speed_of_gust" in ds_meps_sfc else None
t2m  = head24(ds_meps_sfc["air_temperature_2m"]) if "air_temperature_2m" in ds_meps_sfc else None
land = ds_meps_sfc.get("land_area_fraction")
cloud_total = head24(ds_meps_sfc.get("cloud_area_fraction")) if "cloud_area_fraction" in ds_meps_sfc else None
cloud_low   = head24(ds_meps_sfc.get("low_type_cloud_area_fraction")) if "low_type_cloud_area_fraction" in ds_meps_sfc else None
cloud_mid   = head24(ds_meps_sfc.get("medium_type_cloud_area_fraction")) if "medium_type_cloud_area_fraction" in ds_meps_sfc else None
cloud_high  = head24(ds_meps_sfc.get("high_type_cloud_area_fraction")) if "high_type_cloud_area_fraction" in ds_meps_sfc else None
cloud_base  = ds_meps_sfc.get("cloud_base_altitude")
cloud_top   = ds_meps_sfc.get("cloud_top_altitude")

ssrd_int = ds_meps_sfc.get("integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time")
strd_inst= ds_meps_sfc.get("surface_downwelling_longwave_flux_in_air")
strd_int = ds_meps_sfc.get("integral_of_surface_downwelling_longwave_flux_in_air_wrt_time")
ptype    = head24(ds_meps_sfc.get("precipitation_type")) if "precipitation_type" in ds_meps_sfc else None

Upl = ds_meps_pl.get("x_wind_pl"); Vpl = ds_meps_pl.get("y_wind_pl")

# ===== Tab 1: Wind Energy Operator =====
if u10 is not None and v10 is not None:
    # Hub-height wind (approx) + arrows (10m)
    vhub = hub_height_speed(u10, v10, land_frac=land)
    build_png_plus_arrows("Wind Energy • Hub-height wind (m/s)", "t1_wind_hub",
                          scalar_da=vhub, colorscale_name=CMAP["wind"], vmin=0, vmax=30, unit="m/s",
                          u_da=u10, v_da=v10)

    # Gust factor
    if gust is not None:
        gf = gust_factor(gust, u10, v10)
        build_png_plus_arrows("Wind Energy • Gust factor", "t1_wind_gust_factor",
                              scalar_da=gf, colorscale_name=CMAP["gust"], vmin=0, vmax=2.5, unit="")

    # Bulk low-level shear (925–850)
    if (Upl is not None) and (Vpl is not None) and ("pressure" in ds_meps_pl.dims):
        if (925.0 in Upl.pressure.values) and (850.0 in Upl.pressure.values):
            sh_ll = deep_shear(Upl, Vpl, 925.0, 850.0)
            vmax = float(np.nanpercentile(sh_ll.values, 99))
            build_png_plus_arrows("Wind Energy • Shear 925–850 (m/s)", "t1_wind_shear_925_850",
                                  scalar_da=sh_ll, colorscale_name=CMAP["shear"], vmin=0, vmax=vmax, unit="m/s")

    # Capacity factor (0..1) from vhub
    cf = capacity_factor_from_speed(vhub)
    build_png_plus_arrows("Wind Energy • Capacity factor (0–1)", "t1_wind_capacity_factor",
                          scalar_da=cf, colorscale_name="Viridis", vmin=0, vmax=1, unit="")

    # 3h ramp risk (m/s change)
    r3 = ramp_3h(vhub)
    vmax_r = float(np.nanpercentile(r3.values, 99))
    build_png_plus_arrows("Wind Energy • 3h Ramp (m/s)", "t1_wind_ramp3h",
                          scalar_da=r3, colorscale_name="Magma", vmin=0, vmax=vmax_r, unit="m/s")

# ===== Tab 2: Solar Energy Manager =====
if ssrd_int is not None:
    ssrd_rate = rate_from_integral(head24(ssrd_int))
    sw_max = float(np.nanpercentile(ssrd_rate.values, 99))
    build_png_plus_arrows("Solar • Shortwave flux (W/m²)", "t2_solar_ssrd",
                          scalar_da=ssrd_rate, colorscale_name=CMAP["ssrd"], vmin=0, vmax=sw_max, unit="W/m²")
if cloud_total is not None:
    build_png_plus_arrows("Solar • Total cloud (0–1)", "t2_solar_cloud_total",
                          scalar_da=cloud_total.clip(0,1), colorscale_name=CMAP["cloud"], vmin=0, vmax=1, unit="")
if cloud_top is not None:
    ct = head24(cloud_top)
    ct_max = float(np.nanpercentile(ct.values, 99))
    build_png_plus_arrows("Solar • Cloud top (m)", "t2_solar_cloud_top",
                          scalar_da=ct, colorscale_name=CMAP["alt"], vmin=0, vmax=ct_max, unit="m")

# ===== Tab 3: Insurance Risk Analyst =====
precip_rate = precip_rate_pref_components(ds_meps_sfc)
if precip_rate is not None:
    vmax_pr = max(2.0, float(np.nanpercentile(precip_rate.values, 99)))
    build_png_plus_arrows("Insurance • Precip rate (mm/h)", "t3_ins_precip_rate",
                          scalar_da=precip_rate, colorscale_name=CMAP["precip"], vmin=0, vmax=vmax_pr, unit="mm/h")

# 24h accumulation (static single-step)
tp_acc = ds_meps_sfc.get("precipitation_amount_acc")
if tp_acc is not None:
    tp24 = head24(tp_acc)
    acc_24 = (tp24.isel(step=-1) - tp24.isel(step=0)).expand_dims(step=[tp24.step.values[-1]])
    vmax_acc = max(5.0, float(np.nanpercentile(acc_24.values, 99)))
    build_png_plus_arrows("Insurance • 24h Accumulated precip (mm)", "t3_ins_tp24",
                          scalar_da=acc_24, colorscale_name=CMAP["precip"], vmin=0, vmax=vmax_acc, unit="mm")

if gust is not None:
    gmax = gust.max("step").expand_dims(step=[gust.step.values[0]])
    vmax_g = float(np.nanpercentile(gmax.values, 99))
    build_png_plus_arrows("Insurance • Max gust next 24h (m/s)", "t3_ins_max_gust",
                          scalar_da=gmax, colorscale_name=CMAP["gust"], vmin=0, vmax=vmax_g, unit="m/s")

if ptype is not None:
    build_png_plus_arrows("Insurance • Precipitation type", "t3_ins_ptype",
                          scalar_da=ptype.astype("int16"), colorscale_name=None, vmin=int(np.nanmin(ptype.values)),
                          vmax=int(np.nanmax(ptype.values)), unit="", categorical=True)

if t2m is not None:
    freeze_frac = freezing_fraction(ds_meps_sfc["air_temperature_2m"])
    # wrap as single-step
    freeze_map = freeze_frac.expand_dims(step=[head24(ds_meps_sfc["air_temperature_2m"]).step.values[-1]])
    build_png_plus_arrows("Insurance • Freezing fraction next 24h (0–1)", "t3_ins_freeze_frac",
                          scalar_da=freeze_map, colorscale_name=CMAP["freeze"], vmin=0, vmax=1, unit="")

# ===== Tab 4: District Heating Controller =====
if t2m is not None:
    # 2m temperature (°C)
    t2m_c = (t2m - 273.15)
    build_png_plus_arrows("Heating • 2m Temperature (°C)", "t4_heat_t2m",
                          scalar_da=t2m_c, colorscale_name="RdBu_r", vmin=-20, vmax=20, unit="°C")

    # Apparent temperature (wind chill only, simple)
    spd10 = xr.apply_ufunc(np.hypot, u10, v10) if (u10 is not None and v10 is not None) else None
    if spd10 is not None:
        # Standard wind-chill valid if T<=10°C and V>1.34 m/s; else = T
        T = t2m_c
        V = spd10
        wc = 13.12 + 0.6215*T - 11.37*(V**0.16) + 0.3965*T*(V**0.16)
        app = xr.where((T <= 10.0) & (V > 1.34), wc, T)
        build_png_plus_arrows("Heating • Apparent temperature (°C)", "t4_heat_apparent",
                              scalar_da=app, colorscale_name=CMAP["apparent"], vmin=-25, vmax=20, unit="°C")

    # Heating degree hours (static)
    hdh = heating_degree_hours_24h(ds_meps_sfc["air_temperature_2m"])
    hdh_map = hdh.expand_dims(step=[t2m.step.values[-1]])
    vmax_hdh = float(np.nanpercentile(hdh_map.values, 99))
    build_png_plus_arrows("Heating • Heating Degree Hours (°C·h, next 24h)", "t4_heat_hdh",
                          scalar_da=hdh_map, colorscale_name=CMAP["hdh"], vmin=0, vmax=vmax_hdh, unit="°C·h")

# ===== Tab 5: Aviation & Logistics =====
if (Upl is not None) and (Vpl is not None) and ("pressure" in ds_meps_pl.dims):
    if (850.0 in Upl.pressure.values) and (500.0 in Upl.pressure.values):
        sh_dp = deep_shear(Upl, Vpl, 850.0, 500.0)
        vmax = float(np.nanpercentile(sh_dp.values, 99))
        build_png_plus_arrows("Aviation • Deep shear 850–500 (m/s)", "t5_av_shear_850_500",
                              scalar_da=sh_dp, colorscale_name=CMAP["shear"], vmin=0, vmax=vmax, unit="m/s")

    # LLJ max 925–700
    try:
        llj = llj_max(Upl, Vpl)
        vmax_llj = float(np.nanpercentile(llj.values, 99))
        build_png_plus_arrows("Aviation • LLJ max 925–700 (m/s)", "t5_av_llj_max",
                              scalar_da=llj, colorscale_name=CMAP["wind"], vmin=0, vmax=vmax_llj, unit="m/s")
    except Exception:
        pass

if cloud_base is not None:
    ceil_frac = ceiling_risk_fraction(cloud_base, 300.0).expand_dims(step=[head24(cloud_base).step.values[-1]])
    build_png_plus_arrows("Aviation • Ceiling < 300 m fraction (0–1)", "t5_av_ceiling_risk",
                          scalar_da=ceil_frac, colorscale_name="Greys", vmin=0, vmax=1, unit="")

# Optional: liquid-freezing risk proxy (T<0 & precip>0)
if (t2m is not None) and (precip_rate is not None):
    risk = ((head24(ds_meps_sfc["air_temperature_2m"]) < 273.15) & (precip_rate > 0.5)).astype("int16")
    build_png_plus_arrows("Aviation • Freezing precip risk (proxy)", "t5_av_frz_risk",
                          scalar_da=risk, colorscale_name=None, vmin=0, vmax=1, unit="", categorical=True)

print("✅ Thematic JSONs written to ./plot_dump (Tabs 1–5)")
