# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ARG AE_POLICY_VARIANT=normal
ARG AE_ATTACK_MODEL=
ARG AE_ATTACK_BOMB_MARGIN=0.0
ARG AE_ATTACK_MODULE_MODE=hybrid
ARG AE_INSTALL_TORCH=0
ENV AE_POLICY_VARIANT=${AE_POLICY_VARIANT}
ENV AE_ATTACK_MODEL=${AE_ATTACK_MODEL}
ENV AE_ATTACK_BOMB_MARGIN=${AE_ATTACK_BOMB_MARGIN}
ENV AE_ATTACK_MODULE_MODE=${AE_ATTACK_MODULE_MODE}
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$AE_INSTALL_TORCH" = "1" ]; then \
  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
fi

# Copies your source files.
COPY src .
RUN if [ "$AE_INSTALL_TORCH" = "1" ]; then \
  COPY models/ models/; \
fi

# Starts your model server.
CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
