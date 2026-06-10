# The eval-harness image (organiser-internal). It runs the game engine plus the
# 19 random opponents in-process, calls the participant's server over HTTP, and
# prints PASS/FAIL. It is NOT a web server itself.
#
#   build:  docker build -t til26-harness ./server
#   run  :  docker run -e PARTICIPANT_URL=http://host:6700 til26-harness
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "eval_harness.py"]
