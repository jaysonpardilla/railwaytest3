# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy only requirements and install early
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy the rest of the code
COPY . .

# Make sure start script is executable
RUN chmod +x ./start.sh

# Start the server
CMD ["./start.sh"]
