"""
Flask Routes / Blueprints

Registriert alle API-Blueprints.
"""
from flask import Flask


def register_blueprints(app: Flask):
    """Registriert alle Blueprints bei der Flask-App."""

    from src.web.routes.videos import videos_bp
    app.register_blueprint(videos_bp, url_prefix="/api/videos")

    from src.web.routes.labeling import labeling_bp
    app.register_blueprint(labeling_bp)

    from src.web.routes.training import training_bp
    app.register_blueprint(training_bp)

    from src.web.routes.calibration import calibration_bp
    app.register_blueprint(calibration_bp)
