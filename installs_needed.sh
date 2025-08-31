# Install downScaleML from the specified branch
git clone https://github.com/interTwin-eu/downScaleML.git && \
cd downScaleML && \
git checkout openEO_downScaleML && \
pip install .

# Install raster2stac from the test_cube branch
git clone https://gitlab.inf.unibz.it/earth_observation_public/raster-to-stac.git && \
cd raster-to-stac && \
git checkout chunk_auto_bug && \
pip install .

git clone https://github.com/interTwin-eu/openeo-processes-dask.git && \
cd openeo-processes-dask && \
git checkout process/r2s && \
git submodule set-url openeo_processes_dask/specs/openeo-processes https://github.com/interTwin-eu/openeo-processes.git && \
git submodule update --init && \
cd openeo_processes_dask/specs/openeo-processes && \
git checkout downScaleML_processes && \
cd ../../.. && \
pip install .[implementations]

pip install ipython==8.0.0
pip install s3fs