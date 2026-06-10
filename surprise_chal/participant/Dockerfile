# Single-team player server image — THIS is what you build and submit.
#   build:  docker build -t my-team .                        # defaults to AGENT=algo
#   build:  docker build --build-arg AGENT=llm -t my-team .  # bake in the llm agent
#   run  :  docker run -p 6700:6700 -e AGENT=algo my-team
#   run  :  docker run -p 6700:6700 -e AGENT=llm -e OPENROUTER_API_KEY=sk-... my-team
#
# AGENT can be set at BUILD time (--build-arg, baked into the image) or overridden
# at RUN time (-e AGENT=..., which wins). Submitted images are run with no -e
# override, so whatever you bake in here is what the competition uses.
#
# The competition expects your server to listen on port 6700 and answer
# GET /health and POST /observe.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY RULES.md .

# Build-time agent selection: `--build-arg AGENT=llm` bakes ENV AGENT=llm below.
ARG AGENT=algo
ARG OPENROUTER_API_KEY=""

WORKDIR /app/src
ENV PYTHONUNBUFFERED=1 \
    PORT=6700 \
    AGENT=${AGENT} \
    OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
EXPOSE 6700

CMD ["python", "server.py"]
