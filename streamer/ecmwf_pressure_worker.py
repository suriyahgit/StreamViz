from ecmwf.opendata import Client
import os
import xarray as xr
from datetime import datetime, timezone
from raster2stac import Raster2STAC
import logging
import numpy as np
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Client(source="ecmwf", beta=False)

# Get environment variables
DATE = os.environ["DATE"]  # e.g. "2025-08-30"
TIME = os.environ["TIME"]  # "00" | "06" | "12" | "18"

#DATE = "2025-08-30"
#TIME = "06"
STEPS = list(range(0, 46, 3))  # 0..48 by 3h
OUTDIR = "/home/sdhinakaran/task/StreamViz/data/use"

logger.info(f"Processing DATE: {DATE}, TIME: {TIME}")

# Define pressure level retrieval request
req_pl = {
    "date": DATE, "time": TIME, "step": STEPS,
    "type": "fc", "stream": "oper",
    "model": "ifs", "grid": "0.25/0.25",
    "levtype": "pl", "levelist": "925/850/700/500/250",
    "param": ["u", "v", "q", "gh"],
    "class": "od", "domain": "g",
    "target": os.path.join(OUTDIR, f"ifs_pl_{DATE}_{TIME}.grib2"),
}

# Retrieve data
logger.info("Retrieving pressure level data from ECMWF")
client.retrieve(req_pl)
logger.info("Pressure level data retrieval completed")

# Open dataset
logger.info("Opening GRIB2 dataset")
ds = xr.open_dataset(f"/home/sdhinakaran/task/StreamViz/data/use/ifs_pl_{DATE}_{TIME}.grib2")

def clean_coords_pressure(ds):
    """Keep only time, steps, longitude, latitude coordinates"""
    return ds.drop_vars([coord for coord in ds.coords 
                       if coord not in ['time', 'step', 'longitude', 'latitude', 'isobaricInhPa']])

# Clean coordinates
logger.info("Cleaning coordinates")
ds = clean_coords_pressure(ds)

# Convert step coordinate to datetime
logger.info("Converting step coordinate to datetime")
if 'time' in ds.coords and 'step' in ds.coords:
    reference_time = ds.time.values
    ds['step'] = reference_time + ds.step

ds = ds.rio.write_crs("EPSG:4326")

# Process each pressure level
logger.info("Reprojecting each pressure level to EPSG:3035")
levels = ds["isobaricInhPa"].values
reproj_levels = []

for lev in levels:
    logger.info(f"Processing level: {lev}hPa")
    ds_lev = ds.sel(isobaricInhPa=lev)
    ds_lev_3035 = ds_lev.rio.reproject("EPSG:3035")
    ds_lev_3035 = ds_lev_3035.expand_dims({"isobaricInhPa": [float(lev)]})
    reproj_levels.append(ds_lev_3035)

# Concatenate back along isobaricInhPa
logger.info("Concatenating pressure levels")
ds_3035 = xr.concat(reproj_levels, dim="isobaricInhPa")
ds_3035 = ds_3035.sortby("isobaricInhPa")

# Crop to target area
logger.info("Cropping to target area")
cropped_ds = ds_3035.sel(
    x=slice(3602000, 5539203.595174674),
    y=slice(5690158.351704, 3282159.6029571723)
)

# Prepare data for STAC
logger.info("Preparing data for STAC generation")
cropped_ds = cropped_ds.expand_dims('time')
cropped_ds = cropped_ds.to_dataarray(dim="bands")

# Set attributes
cropped_ds.attrs["crs"] = "EPSG:3035"
cropped_ds.attrs["proj:epsg"] = 3035
cropped_ds.attrs["spatial_ref"] = "EPSG:3035"
cropped_ds.attrs["forecast_reference_time"] = str(cropped_ds.step.values[0])

# Define task metadata
task_SkyFora = {
    "name": "Suriyah Dhinakaran",
    "url": "https://github.com/suriyahgit/StreamViz",
    "roles": ["Creator"],
}

# Generate ZARR STAC
logger.info("Generating ZARR STAC for pressure levels")
output_folder = f"/home/sdhinakaran/task/StreamViz/data/r2s_dump/IFS_DET_PRESSURE_{DATE}_{TIME}z"

rs2stac = Raster2STAC(
    data=cropped_ds,
    write_collection_assets=True,
    collection_id="ECMWF_IFS_DET_PRESSURE_LEVELS_FORECAST_0_25_DEG",
    description="The ECMWF Integrated Forecasting System (IFS) deterministic forecast is a global numerical weather prediction model...",
    license="Apache-2.0",
    keywords=["SkyFora", "Forecast", "Zarr", "IFS"],
    collection_url="http://localhost:8081/collections/",
    output_folder=output_folder,
    providers=[task_SkyFora],
    s3_upload=False,
).generate_zarr_stac(item_id=f"IFS_DET_PRESSURE_{DATE}_{TIME}z")

# Post STAC items
logger.info("Posting STAC items to server")
stac_items = []
with open(f"{output_folder}/inline_items.csv", "r") as f:
    stac_items = f.readlines()

for it in stac_items:
    stac_data_to_post = json.loads(it)
    requests.post("http://localhost:8081/collections/ECMWF_IFS_DET_PRESSURE_LEVELS_FORECAST_0_25_DEG/items", json=stac_data_to_post)

logger.info(f"IFS_DET_PRESSURE_PIPELINE FOR the slot {DATE}_{TIME}z Completed!")