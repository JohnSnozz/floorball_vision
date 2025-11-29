"""
Flask Konfiguration

Lädt Einstellungen aus .env und definiert Config-Klassen.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    """Basis-Konfiguration für alle Umgebungen."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    # Celery
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # Pfade
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    MODELS_PATH = os.getenv("MODELS_PATH", "./models")
    VIDEOS_PATH = os.path.join(DATA_PATH, "videos")
    FRAMES_PATH = os.path.join(DATA_PATH, "frames")

    # Upload
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5 GB
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

    # Label Studio
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")


class DevelopmentConfig(BaseConfig):
    """Entwicklungs-Konfiguration."""

    DEBUG = True
    TESTING = False


class ProductionConfig(BaseConfig):
    """Produktions-Konfiguration."""

    DEBUG = False
    TESTING = False


class TestingConfig(BaseConfig):
    """Test-Konfiguration."""

    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


# Config auswählen basierend auf FLASK_ENV
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    """Gibt die aktuelle Konfiguration zurück."""
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
