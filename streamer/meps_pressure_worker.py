import pyproj
import xarray as xr
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

logger.info(f"Processing DATE: {DATE}, TIME: {TIME}")

# Reformat DATE to YYYYMMDD
ymd = DATE.replace("-", "")
url = f"https://thredds.met.no/thredds/dodsC/mepslatest/meps_det_2_5km_{ymd}T{TIME}Z.ncml"

logger.info(f"Opening dataset from URL: {url}")

# Define variables needed and open dataset
vars_needed_pl = [
    "x_wind_pl", "y_wind_pl", "air_temperature_pl", 
    "specific_humidity_pl", "geopotential_pl"
]

ds = xr.open_dataset(url, chunks={"time": 4})[vars_needed_pl]
ds = ds.isel(time=slice(0, 48))
ds = ds.sel(pressure=["925", "850", "700", "500", "250"])

logger.info("Dataset loaded and filtered")

# Clean coordinates
logger.info("Resetting coordinates")
ds = ds.reset_coords(("latitude", "longitude"), drop=True)

# Set CRS
logger.info("Setting CRS")
crs = pyproj.CRS.from_cf({
    "grid_mapping_name": "lambert_conformal_conic",
    "standard_parallel": [63.3, 63.3],
    "longitude_of_central_meridian": 15.0,
    "latitude_of_projection_origin": 63.3,
    "earth_radius": 6371000.0,
})

ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
ds = ds.rio.write_crs(crs)

# Load data into memory
logger.info("Loading data into memory")
store = ds.load()

# Process each pressure level
logger.info("Reprojecting each pressure level to EPSG:3035")
levels = store["pressure"].values
reproj_levels = []

for lev in levels:
    logger.info(f"Processing pressure level: {lev}hPa")
    ds_lev = store.sel(pressure=lev)
    ds_lev_3035 = ds_lev.rio.reproject("EPSG:3035")
    ds_lev_3035 = ds_lev_3035.expand_dims({"pressure": [float(lev)]})
    reproj_levels.append(ds_lev_3035)

# Concatenate back along pressure dimension
logger.info("Concatenating pressure levels")
ds_3035 = xr.concat(reproj_levels, dim="pressure")
ds_3035 = ds_3035.sortby("pressure")

# Crop to target area
logger.info("Cropping to target area")
merged_cropped = ds_3035.sel(
    x=slice(3602000, 5539203.595174674),
    y=slice(5690158.351704, 3282159.6029571723)
)

# Set attributes and prepare for STAC
logger.info("Setting attributes and preparing data for STAC")
merged_cropped.attrs["crs"] = "EPSG:3035"
merged_cropped.attrs["proj:epsg"] = 3035
merged_cropped.attrs["spatial_ref"] = "EPSG:3035"
merged_cropped.attrs["forecast_reference_time"] = str(ds.time.values[0])
merged_cropped = merged_cropped.rename({"time": "step"})

first_step_time = merged_cropped["step"].isel(step=0).values
merged_cropped = merged_cropped.assign_coords(time=first_step_time)
merged_cropped = merged_cropped.expand_dims('time')
merged_cropped = merged_cropped.to_dataarray(dim="bands")

# Define task metadata
task_SkyFora = {
    "name": "Suriyah Dhinakaran",
    "url": "https://github.com/suriyahgit/StreamViz",
    "roles": ["Creator"],
}

# Generate ZARR STAC
logger.info("Generating ZARR STAC")
output_folder = f"/home/sdhinakaran/task/StreamViz/data/r2s_dump/MEPS_DET_PRESSURE_{ymd}T{TIME}Z"

rs2stac = Raster2STAC(
    data=merged_cropped,
    write_collection_assets=True,
    collection_id="MEPS_DET_PRESSURE_2_5KMS",
    description="MEPS_DET_PRESSURE: The MetCoOp Ensemble Prediction System (MEPS) deterministic pressure-level model...",
    license="Apache-2.0",
    keywords=["SkyFora", "Forecast", "Zarr", "MEPS"],
    collection_url="http://localhost:8081/collections/",
    output_folder=output_folder,
    providers=[task_SkyFora],
    s3_upload=False,
).generate_zarr_stac(item_id=f"MEPS_DET_PRESSURE_{ymd}T{TIME}Z")

# Post STAC items
logger.info("Posting STAC items to server")
stac_items = []
with open(f"{output_folder}/inline_items.csv", "r") as f:
    stac_items = f.readlines()

for it in stac_items:
    stac_data_to_post = json.loads(it)
    requests.post("http://localhost:8081/collections/MEPS_DET_PRESSURE_2_5KMS/items", json=stac_data_to_post)

logger.info(f"MEPS_DET_PRESSURE_PIPELINE FOR the slot {DATE}_{TIME}z Completed!")