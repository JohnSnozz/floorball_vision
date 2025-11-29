"""
Flask Routes / Blueprints

Registriert alle API-Blueprints.
"""
from flask import Flask


def register_blueprints(app: Flask):
    """Registriert alle Blueprints bei der Flask-App."""

    from src.web.routes.videos import videos_bp
    app.register_blueprint(videos_bp, url_prefix="/api/videos")
