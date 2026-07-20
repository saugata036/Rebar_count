# Simple Dockerfile for Rebar Analyzer FastAPI app
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps required by opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Ensure outputs directory exists
RUN mkdir -p /app/outputs

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
