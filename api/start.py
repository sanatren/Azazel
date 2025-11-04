#!/usr/bin/env python3
"""
Entrypoint script for Railway deployment
Handles PORT environment variable properly
"""
import os
import sys

if __name__ == "__main__":
    port = os.getenv("PORT", "8000")
    host = "0.0.0.0"

    print(f"Starting Uvicorn on {host}:{port}")

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=int(port),
        log_level="info"
    )
