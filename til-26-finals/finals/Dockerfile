FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src .

CMD ["python", "participant_server.py"]