# Production Multi-Stage Dockerfile for Retail Inventory Reorder Assistant
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application source code and data
COPY . .

# Ensure pipeline data and model artifacts exist
RUN python run_pipeline.py

# Expose ports: 8501 for Streamlit Dashboard, 8000 for FastAPI REST API
EXPOSE 8501
EXPOSE 8000

# Default command: launch the interactive Streamlit Dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
