"""
Configuration settings for the FastAPI application
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API Keys (from environment variables)
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CSE_ID: Optional[str] = None

    # Supabase
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:8501", "http://localhost:3000", "*"]

    # Timeouts
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    STREAM_TIMEOUT: int = 600  # 10 minutes

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()