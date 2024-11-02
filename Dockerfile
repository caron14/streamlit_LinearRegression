FROM python:3.10-slim

WORKDIR /opt
COPY requirements.txt /opt/

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

WORKDIR /work
