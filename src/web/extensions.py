"""
Flask Extensions

Zentrale Instanzen für Flask-Extensions.
Wird von app.py initialisiert und von anderen Modulen importiert.
"""
import os
from flask_sqlalchemy import SQLAlchemy
from celery import Celery
from dotenv import load_dotenv

# .env laden für Celery Konfiguration
load_dotenv()

# SQLAlchemy Instanz
db = SQLAlchemy()

# Celery Instanz mit Redis Konfiguration
celery = Celery(
    __name__,
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

# Celery Konfiguration
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Zurich",
    enable_utc=True,
    imports=["src.processing.tasks"],  # Tasks automatisch laden
)
