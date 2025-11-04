# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from api directory
COPY api/requirements.txt .

# Install Python dependencies - optimized for size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache
    rm -rf /root/.cache/pip/*

# Copy Bot modules (needed for API)
COPY Bot /app/Bot

# Copy API code
COPY api /app/api

# Copy and set permissions for entrypoint script
COPY api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set Python path
ENV PYTHONPATH=/app

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Run the application using bash entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
