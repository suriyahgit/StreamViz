FROM mambaorg/micromamba:latest

USER root
ENV MAMBA_DOCKERFILE_ACTIVATE=1 MAMBA_ROOT_PREFIX=/opt/conda

RUN apt-get update && apt-get install -y \
    git curl netcat-openbsd ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown -R mambauser:mambauser /app
USER mambauser

COPY --chown=mambauser:mambauser environment.yml .
COPY --chown=mambauser:mambauser test_requirements.txt .

RUN micromamba env create -f environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/envs/StreamViz/bin:$PATH

# Optional: raster2stac install (if you use it)
RUN git clone https://gitlab.inf.unibz.it/earth_observation_public/raster-to-stac.git && \
    cd raster-to-stac && \
    git checkout chunk_auto_bug && \
    pip install .

RUN pip install -r test_requirements.txt && pip install s3fs requests


RUN pip install \
  stac-fastapi.api==6.0.0 \
  stac-fastapi.extensions==6.0.0 \
  stac-fastapi.types==6.0.0 \
  stac-fastapi.pgstac==6.0.0 \
  fastapi>=0.104.0 \
  uvicorn>=0.24.0 \
  asyncpg>=0.28.0 \
  pypgstac


# --- STAC FastAPI (updated working combination) ---
#RUN pip install \
#    stac-fastapi.api==2.4.13 \
#    stac-fastapi.extensions==2.4.13 \
#    stac-fastapi.pgstac==2.4.13 \
#    stac-fastapi.types==2.4.13 \
#    fastapi>=0.104.0,<0.105.0 \
#    uvicorn>=0.24.0 \
#    psycopg[binary]>=3.1.10 \
#    pgstac==0.7.9

# Alternative: If you prefer the newer asyncpg-based version
#RUN pip install \
#    stac-fastapi.api==2.5.0 \
#    stac-fastapi.extensions==2.5.0 \
#    stac-fastapi.pgstac==2.5.0 \
#    stac-fastapi.types==2.5.0 \
#    fastapi>=0.104.0 \
#    uvicorn>=0.24.0 \
#    asyncpg>=0.28.0 \
#    pgstac==0.9.8

ENV STAC_API_PORT=8081 \
    POSTGRES_HOST=db \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=stac \
    POSTGRES_USER=stac \
    POSTGRES_PASSWORD=stac

COPY --chown=mambauser:mambauser start-stac.sh /usr/local/bin/start-stac.sh
RUN chmod +x /usr/local/bin/start-stac.sh

EXPOSE 8081
CMD ["bash"]