# =============================================================================
# FULL BUILD — used to create the base image from scratch (e.g. v0.2)
# To rebuild from scratch, replace the FROM below with this block:
#
# FROM registry.rcp.epfl.ch/rcp-courses/2026-ee-559:latest
#
# RUN apt update && apt install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt update && apt install -y python3.12 python3.12-venv python3.12-dev && \
#     rm -r /var/lib/apt/lists/*
#
# RUN python3.12 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# RUN pip install --upgrade pip
# =============================================================================

# LIGHTWEIGHT UPDATE — builds on top of the previous image (faster push)
# Always pass the PREVIOUS version explicitly (do NOT use latest — circular dependency risk).
# Example to build v0.4 from v0.3:
#   docker build --platform linux/amd64 \
#                --build-arg BASE=registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.3 \
#                -t registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.4 .
#   docker push registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.4
ARG BASE=registry.rcp.epfl.ch/ee-559-guenzi/my-toolbox:v0.2
FROM ${BASE}

# Copy and install requirements (only new/changed deps are added)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
