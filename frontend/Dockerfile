# Simple Python-based HTTP server for static files
FROM python:3.12-slim

WORKDIR /app

# Copy frontend files
COPY . /app

# Expose port
EXPOSE 8080

# Run the server
CMD ["python", "server.py"]