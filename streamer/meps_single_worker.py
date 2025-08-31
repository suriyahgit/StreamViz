import xarray as xr
import pyproj
from datetime import datetime, timezone
from raster2stac import Raster2STAC
import logging
import os
import numpy as np
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
DATE = os.environ["DATE"]  # e.g. "2025-08-30"
TIME = os.environ["TIME"]  # "00" | "06" | "12" | "18"

#DATE = "2025-08-31" 
#TIME = "00"

logger.info(f"Processing DATE: {DATE}, TIME: {TIME}")

# Reformat DATE to YYYYMMDD
ymd = DATE.replace("-", "")
url = f"https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_{ymd}T{TIME}Z.ncml"

logger.info(f"Opening dataset from URL: {url}")

# Define variable groups by height level
height0_vars = [
    "land_area_fraction", "cloud_area_fraction", "low_type_cloud_area_fraction",
    "medium_type_cloud_area_fraction", "high_type_cloud_area_fraction",
    "precipitation_amount_acc", "snowfall_amount_acc",
    "integral_of_rainfall_amount_wrt_time", "integral_of_graupelfall_amount_wrt_time",
    "rainfall_amount", "snowfall_amount", "graupelfall_amount", "precipitation_type",
    "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
    "surface_downwelling_longwave_flux_in_air",
    "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
]

height7_vars = ["x_wind_10m", "y_wind_10m", "wind_speed_of_gust"]
surface_vars = ["cloud_base_altitude", "cloud_top_altitude"]
height1_vars = ["air_temperature_2m"]
height_above_msl_vars = ["air_pressure_at_sea_level"]
height4_vars = ["cloud_base_altitude_z"]

# Load and process each variable group
logger.info("Loading height0 variables")
ds = xr.open_dataset(url)[height0_vars]
ds = ds.isel(time=slice(0, 24), height0=0)
store_1 = ds.load()

logger.info("Loading height7 variables")
store_2 = xr.open_dataset(url)[height7_vars]
store_2 = store_2.isel(time=slice(0, 24), height7=0)
store_2 = store_2.load()

logger.info("Loading surface variables")
store_3 = xr.open_dataset(url)[surface_vars]
store_3 = store_3.isel(time=slice(0, 24), surface=0)
store_3 = store_3.load()

logger.info("Loading height1 variables")
var1 = xr.open_dataset(url)[height1_vars]
var1 = var1.isel(time=slice(0, 24), height1=0)
var1 = var1.load()

logger.info("Loading height_above_msl variables")
var2 = xr.open_dataset(url)[height_above_msl_vars]
var2 = var2.isel(time=slice(0, 24), height_above_msl=0)
var2 = var2.load()

logger.info("Loading height4 variables")
var3 = xr.open_dataset(url)[height4_vars]
var3 = var3.isel(time=slice(0, 24), height4=0)
var3 = var3.load()

# Merge all datasets
logger.info("Merging all datasets")
merged = xr.merge([store_1, store_2, store_3, var1, var2, var3])

# Clean coordinates
logger.info("Cleaning coordinates")
merged = merged.reset_coords(("latitude", "longitude"), drop=True)

def clean_coords(ds):
    """Keep only time, steps, longitude, latitude coordinates"""
    return ds.drop_vars([coord for coord in ds.coords 
                       if coord not in ['time', 'x', 'y']])

merged = clean_coords(merged)

# Set CRS and reproject
logger.info("Setting CRS and reprojecting")
crs = pyproj.CRS.from_cf({
    "grid_mapping_name": "lambert_conformal_conic",
    "standard_parallel": [63.3, 63.3],
    "longitude_of_central_meridian": 15.0,
    "latitude_of_projection_origin": 63.3,
    "earth_radius": 6371000.0,
})

merged = merged.rio.set_spatial_dims(x_dim="x", y_dim="y")
merged = merged.rio.write_crs(crs)
merged = merged.rio.reproject("EPSG:4326")

# Crop to target area
logger.info("Cropping to target area")
store = merged.sel(
    x=slice(0, 32),
    y=slice(72, 51.5)
)

# Set attributes and prepare for STAC
logger.info("Setting attributes and preparing data for STAC")
store.attrs["crs"] = "EPSG:4326"
store.attrs["proj:epsg"] = 4326
store.attrs["spatial_ref"] = "EPSG:4326"
store.attrs["forecast_reference_time"] = str(ds.time.values[0])
store = store.rename({"time": "step"})

first_step_time = store["step"].isel(step=0).values
store = store.assign_coords(time=first_step_time)
store = store.expand_dims('time')
store = store.to_dataarray(dim="bands")

# Define task metadata
task_SkyFora = {
    "name": "Suriyah Dhinakaran",
    "url": "https://github.com/suriyahgit/StreamViz",
    "roles": ["Creator"],
}

# Generate ZARR STAC
logger.info("Generating ZARR STAC")
output_folder = f"/home/sdhinakaran/task/StreamViz/data/r2s_dump/MEPS_DET_SINGLE_{ymd}T{TIME}Z"

rs2stac = Raster2STAC(
    data=store,
    write_collection_assets=True,
    collection_id="MEPS_DET_SINGLE_2_5KMS",
    description="MEPS_DET_SINGLE: The MetCoOp Ensemble Prediction System (MEPS) deterministic single-level model...",
    license="Apache-2.0",
    keywords=["SkyFora", "Forecast", "Zarr", "MEPS"],
    collection_url="http://localhost:8081/collections/",
    output_folder=output_folder,
    providers=[task_SkyFora],
    s3_upload=False,
).generate_zarr_stac(item_id=f"MEPS_DET_SINGLE_{ymd}T{TIME}Z")

# Post STAC items
logger.info("Posting STAC items to server")
stac_items = []
with open(f"{output_folder}/inline_items.csv", "r") as f:
    stac_items = f.readlines()

for it in stac_items:
    stac_data_to_post = json.loads(it)
    requests.post("http://localhost:8081/collections/MEPS_DET_SINGLE_2_5KMS/items", json=stac_data_to_post)

logger.info(f"MEPS_DET_SINGLE_PIPELINE FOR the slot {DATE}_{TIME}z Completed!")