"""
Flask Application Factory

Erstellt und konfiguriert die Flask-Anwendung.
"""
import os
from flask import Flask

# Extensions aus separatem Modul importieren
from src.web.extensions import db, celery


def create_app(config_class=None):
    """
    Application Factory Pattern.

    Args:
        config_class: Optionale Konfigurationsklasse

    Returns:
        Flask: Konfigurierte Flask-Anwendung
    """
    app = Flask(__name__)

    # Konfiguration laden
    if config_class is None:
        from src.web.config import get_config
        config_class = get_config()

    app.config.from_object(config_class)

    # Sicherstellen dass Verzeichnisse existieren
    os.makedirs(app.config.get("VIDEOS_PATH", "data/videos"), exist_ok=True)
    os.makedirs(app.config.get("FRAMES_PATH", "data/frames"), exist_ok=True)

    # SQLAlchemy initialisieren
    db.init_app(app)

    # Celery konfigurieren
    celery.conf.update(
        broker_url=app.config["CELERY_BROKER_URL"],
        result_backend=app.config["CELERY_RESULT_BACKEND"],
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="Europe/Zurich",
        enable_utc=True,
    )

    # Celery Task Context
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    # Blueprints registrieren
    from src.web.routes import register_blueprints
    register_blueprints(app)

    # Index Route
    @app.route("/")
    def index():
        from flask import render_template
        return render_template("index.html")

    # Health Check
    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app


# Für direktes Ausführen
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
