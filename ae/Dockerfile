# Dockerfile for building the AE image.


# The base image. A slim Python base keeps the image small and evaluation fast.
# The learned RL policy (rl_policy.py) is a ~1.1M-param net whose per-step
# inference is sub-millisecond on CPU, so no GPU is needed — torch (CPU build)
# is installed from requirements.txt.
#
# If you ever need GPU inference (much larger nets, batched eval), swap the base
# to the CUDA image below — it already ships torch, so also remove the torch /
# --extra-index-url lines from requirements.txt:
#   FROM nvcr.io/nvidia/pytorch:25.11-py3
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .
COPY models/ models/

# Starts your model server.
CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
