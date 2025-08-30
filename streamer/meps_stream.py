import os
import xarray as xr
from datetime import datetime, timezone
from raster2stac import Raster2STAC
import logging
import numpy as np
import requests
import json
import pyproj

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_coords_meps(ds):
    """Keep only time, steps, longitude, latitude coordinates"""
    return ds.drop_vars([coord for coord in ds.coords 
                       if coord not in ['time', 'x', 'y']])

task_SkyFora = {
    "name": "Suriyah Dhinakaran",
    "url": "https://github.com/suriyahgit/StreamViz",
    "roles": ["Creator"],
}

# Get environment variables
DATE = os.environ["DATE"]  # e.g. "2025-08-30"
TIME = os.environ["TIME"]  # "00" | "06" | "12" | "18"

#DATE = "2025-08-30"
#TIME = "06"

logger.info(f"Processing DATE: {DATE}, TIME: {TIME}")

# Reformat DATE to YYYYMMDD
ymd = DATE.replace("-", "")
url = f"https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_{ymd}T{TIME}Z.ncml"

logger.info(f"Opening dataset from URL: {url}")
ds = xr.open_dataset(url, chunks={"time": 4})

# Extract variables
logger.info("Extracting variables from dataset")
air_temperature_2m = ds["air_temperature_2m"].isel(height1=0).isel(time=slice(0, 48))
x_wind_10m = ds["x_wind_10m"].isel(height7=0).isel(time=slice(0, 48))
y_wind_10m = ds["y_wind_10m"].isel(height7=0).isel(time=slice(0, 48))
wind_speed_of_gust = ds["wind_speed_of_gust"].isel(height7=0).isel(time=slice(0, 48))
air_pressure_at_sea_level = ds["air_pressure_at_sea_level"].isel(height_above_msl=0).isel(time=slice(0, 48))
land_area_fraction = ds["land_area_fraction"].isel(height0=0).isel(time=slice(0, 48))

# Merge datasets
logger.info("Merging datasets")
merged = xr.merge([
    air_temperature_2m, 
    x_wind_10m, 
    y_wind_10m, 
    wind_speed_of_gust, 
    air_pressure_at_sea_level, 
    land_area_fraction
])
merged = merged.reset_coords(("latitude", "longitude"), drop=True)
merged = clean_coords_meps(merged)

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
merged = merged.rio.reproject("EPSG:3035")

# Crop and prepare store
logger.info("Cropping and preparing data store")
merged_cropped = merged.sel(
    x=slice(3602000, 5539203.595174674),
    y=slice(5690158.351704, 3282159.6029571723)
)
store = merged_cropped
store.attrs["crs"] = "EPSG:3035"
store.attrs["proj:epsg"] = 3035
store.attrs["spatial_ref"] = "EPSG:3035"
store.attrs["forecast_reference_time"] = str(ds.time.values[0])
store = store.rename({"time": "step"})
first_step_time = store["step"].isel(step=0).values
store = store.assign_coords(time=first_step_time)
store = store.expand_dims('time')
store = store.to_dataarray(dim="bands")

# Generate ZARR STAC
logger.info("Generating ZARR STAC")
output_folder = f"/home/sdhinakaran/task/StreamViz/data/r2s_dump/MEPS_DET_SINGLE_{ymd}T{TIME}Z"

rs2stac = Raster2STAC(
    data=store,
    write_collection_assets=True,
    collection_id="MEPS_DET_SINGLE_2.5KMS",
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
    requests.post("http://localhost:8081/collections/MEPS_DET_SINGLE_2.5KMS/items", json=stac_data_to_post)

logger.info(f"MEPS_SINGLE_PIPELINE FOR the slot {DATE}_{TIME}z Completed!")