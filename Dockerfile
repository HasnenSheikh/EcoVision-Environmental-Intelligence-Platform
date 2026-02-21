FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Prophet / cmdstanpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip --quiet && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create runtime directories
RUN mkdir -p data models

# Train models at build time
RUN python train_models.py

# Hugging Face Spaces runs on port 7860
ENV PORT=7860
ENV FLASK_ENV=production

EXPOSE 7860

# Start gunicorn
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:7860"]
