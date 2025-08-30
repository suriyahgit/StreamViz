from ecmwf.opendata import Client
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

client = Client(source="ecmwf", beta=False)

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

STEPS = list(range(0, 46, 3))  # 0..48 by 3h
OUTDIR = "/home/sdhinakaran/task/StreamViz/data/use"

logger.info(f"Processing DATE: {DATE}, TIME: {TIME}")

# Define retrieval requests
req_sl = {
    "date": DATE, "time": TIME, "step": STEPS,
    "type": "fc", "stream": "oper",
    "param": ["10u", "10v", "10fg", "2t", "sp"],
    "target": os.path.join(OUTDIR, f"ifs_sl_{DATE}_{TIME}.grib2"),
}
req_slc = {
    "date": DATE, "time": TIME, "step": STEPS,
    "type": "fc", "stream": "oper",
    "param": ["100u", "100v", "msl"],
    "target": os.path.join(OUTDIR, f"ifs_slc_{DATE}_{TIME}.grib2"),
}
req_2t = {
    "date": DATE, "time": TIME, "step": STEPS,
    "type": "fc", "stream": "oper",
    "param": ["2t"],
    "target": os.path.join(OUTDIR, f"ifs_2t_{DATE}_{TIME}.grib2"),
}

# Retrieve data
logger.info("Retrieving data from ECMWF")
client.retrieve(req_sl)
client.retrieve(req_slc)
client.retrieve(req_2t)
logger.info("Data retrieval completed")

# Open datasets
logger.info("Opening GRIB2 datasets")
ds1 = xr.open_dataset(f"/home/sdhinakaran/task/StreamViz/data/use/ifs_sl_{DATE}_{TIME}.grib2")
ds2 = xr.open_dataset(f"/home/sdhinakaran/task/StreamViz/data/use/ifs_slc_{DATE}_{TIME}.grib2")
ds3 = xr.open_dataset(f"/home/sdhinakaran/task/StreamViz/data/use/ifs_2t_{DATE}_{TIME}.grib2")

def clean_coords(ds):
    """Keep only time, steps, longitude, latitude coordinates"""
    return ds.drop_vars([coord for coord in ds.coords 
                       if coord not in ['time', 'step', 'longitude', 'latitude']])

# Clean coordinates and merge datasets
logger.info("Cleaning coordinates and merging datasets")
ds1 = clean_coords(ds1)
ds2 = clean_coords(ds2)
ds3 = clean_coords(ds3)
ds = xr.merge([ds1, ds2, ds3])

# Convert step coordinate to datetime
logger.info("Converting step coordinate to datetime")
if 'time' in ds.coords and 'step' in ds.coords:
    reference_time = ds.time.values
    ds['step'] = reference_time + ds.step

# Reproject and crop
logger.info("Reprojecting to EPSG:3035")
ds = ds.rio.write_crs("EPSG:4326")
ds_reproj = ds.rio.reproject("EPSG:3035")

logger.info("Cropping to target area")
ds_cropped = ds_reproj.sel(
    x=slice(3602000, 5539203.595174674),
    y=slice(5690158.351704, 3282159.6029571723)
)

# Prepare data for STAC
logger.info("Preparing data for STAC generation")
ds_cropped = ds_cropped.drop_vars("time")
first_step_time = ds_cropped["step"].isel(step=0).values
ds_cropped = ds_cropped.assign_coords(time=first_step_time)
ds_cropped = ds_cropped.expand_dims('time')
ds_cropped = ds_cropped.to_dataarray(dim="bands")

# Set attributes
ds_cropped.attrs["crs"] = "EPSG:3035"
ds_cropped.attrs["proj:epsg"] = 3035
ds_cropped.attrs["spatial_ref"] = "EPSG:3035"
ds_cropped.attrs["forecast_reference_time"] = str(ds_cropped.step.values[0])

# Generate ZARR STAC
logger.info("Generating ZARR STAC")
output_folder = f"/home/sdhinakaran/task/StreamViz/data/r2s_dump/IFS_DET_SINGLE_{DATE}_{TIME}z"

rs2stac = Raster2STAC(
    data=ds_cropped,
    write_collection_assets=True,
    collection_id="ECMWF_IFS_DET_FORECAST_0_25_DEG",
    description="The ECMWF Integrated Forecasting System (IFS) deterministic forecast is a global numerical weather prediction model...",
    license="Apache-2.0",
    keywords=["SkyFora", "Forecast", "Zarr", "IFS"],
    collection_url="http://localhost:8081/collections/",
    output_folder=output_folder,
    providers=[task_SkyFora],
    s3_upload=False,
).generate_zarr_stac(item_id=f"IFS_DET_SINGLE_{DATE}_{TIME}z")

# Post STAC items
logger.info("Posting STAC items to server")
stac_items = []
with open(f"{output_folder}/inline_items.csv", "r") as f:
    stac_items = f.readlines()

for it in stac_items:
    stac_data_to_post = json.loads(it)
    requests.post("http://localhost:8081/collections/ECMWF_IFS_DET_FORECAST_0_25_DEG/items", json=stac_data_to_post)

logger.info(f"IFS_DET_SINGLE_PIPELINE FOR the slot {DATE}_{TIME}z Completed!")