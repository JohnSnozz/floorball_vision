"""
YOLO ML Backend für Label Studio Pre-Annotation.

Startet einen Flask-Server der YOLO-Vorhersagen für Label Studio bereitstellt.
Label Studio kann sich mit diesem Backend verbinden und automatisch
Pre-Annotationen für neue Bilder generieren.

Basiert auf: https://github.com/HumanSignal/label-studio-ml-backend
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globale Variablen
_model = None
_model_path = None
_class_names = {}


def get_model():
    """Lazy-load des YOLO Modells."""
    global _model, _model_path
    if _model is None and _model_path:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {_model_path}")
            _model = YOLO(_model_path)
            logger.info(f"Model loaded successfully. Classes: {_model.names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model


def create_app(model_path: str, confidence: float = 0.25) -> Flask:
    """
    Erstellt die Flask-App für das ML Backend.

    Args:
        model_path: Pfad zum YOLO .pt Modell
        confidence: Confidence-Threshold für Detektionen

    Returns:
        Flask App
    """
    global _model_path
    _model_path = model_path

    app = Flask(__name__)
    CORS(app)

    app.config['CONFIDENCE'] = confidence
    app.config['MODEL_PATH'] = model_path

    @app.route('/health', methods=['GET'])
    def health():
        """Health-Check Endpoint."""
        model = get_model()
        return jsonify({
            "status": "ok",
            "model_loaded": model is not None,
            "model_path": str(_model_path),
            "classes": model.names if model else {}
        })

    @app.route('/setup', methods=['POST'])
    def setup():
        """
        Label Studio ruft diesen Endpoint auf um das Backend zu konfigurieren.
        Gibt die unterstützten Label-Typen zurück.
        """
        model = get_model()
        return jsonify({
            "model_version": os.path.basename(str(_model_path)),
            "labels": list(model.names.values()) if model else []
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Hauptendpoint für Vorhersagen.
        Label Studio sendet Tasks (Bilder) und erwartet Annotationen zurück.
        """
        try:
            data = request.json
            logger.info(f"Predict request received. Data keys: {list(data.keys()) if data else 'None'}")
            logger.info(f"Full request data: {json.dumps(data, indent=2, default=str)[:2000]}")
            tasks = data.get('tasks', [])

            if not tasks:
                return jsonify({"results": []})

            model = get_model()
            if not model:
                return jsonify({"error": "Model not loaded"}), 500

            confidence = app.config['CONFIDENCE']
            results = []

            for task in tasks:
                task_id = task.get('id')
                # Label Studio kann verschiedene Feldnamen verwenden
                task_data = task.get('data', {})
                image_url = task_data.get('image') or task_data.get('url') or task_data.get('img') or ''
                logger.info(f"Task {task_id} data keys: {list(task_data.keys())}, image_url: {image_url[:100] if image_url else 'EMPTY'}")

                # Label Studio URL zu lokalem Pfad konvertieren
                image_path = _resolve_image_path(image_url)

                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_url} -> {image_path}")
                    results.append({
                        "result": [],
                        "score": 0,
                        "model_version": os.path.basename(str(_model_path))
                    })
                    continue

                # YOLO Inference
                try:
                    predictions = model.predict(
                        image_path,
                        conf=confidence,
                        verbose=False
                    )

                    annotations = _convert_to_label_studio(predictions[0], model.names)

                    results.append({
                        "result": annotations,
                        "score": _calculate_avg_score(predictions[0]),
                        "model_version": os.path.basename(str(_model_path))
                    })

                    logger.info(f"Task {task_id}: {len(annotations)} detections")

                except Exception as e:
                    logger.error(f"Prediction failed for {image_path}: {e}")
                    results.append({
                        "result": [],
                        "score": 0,
                        "model_version": os.path.basename(str(_model_path))
                    })

            return jsonify({"results": results})

        except Exception as e:
            logger.error(f"Predict endpoint error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route('/webhook', methods=['POST'])
    def webhook():
        """Webhook für Label Studio Events (optional)."""
        data = request.json
        logger.info(f"Webhook received: {json.dumps(data, indent=2, default=str)[:1000]}")
        return jsonify({"status": "ok"})

    @app.route('/versions', methods=['GET'])
    def versions():
        """Gibt verfügbare Modell-Versionen zurück."""
        model = get_model()
        return jsonify({
            "versions": [os.path.basename(str(_model_path))] if model else []
        })

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint für Health Check."""
        return jsonify({"status": "ok", "message": "YOLO ML Backend for Label Studio"})

    # Log alle eingehenden Requests
    @app.before_request
    def log_request():
        logger.info(f"Request: {request.method} {request.path}")

    return app


def _resolve_image_path(url: str) -> Optional[str]:
    """
    Konvertiert Label Studio URL zu lokalem Dateipfad.

    Label Studio URLs können verschiedene Formate haben:
    - /data/local-files/?d=path/to/file.jpg
    - /data/upload/project/file.jpg
    - http://localhost:8080/data/...
    - Absolute Pfade
    """
    if not url:
        return None

    from urllib.parse import unquote, urlparse

    # URL dekodieren
    url = unquote(url)

    # Bereits absoluter Pfad
    if os.path.isabs(url) and os.path.exists(url):
        return url

    # HTTP URL - versuche zu fetchen und temporär zu speichern
    if url.startswith('http://') or url.startswith('https://'):
        try:
            import requests
            import tempfile

            # Parse URL
            parsed = urlparse(url)

            # Label Studio local-files Format in URL
            if '/data/local-files/' in url and 'd=' in url:
                # Extrahiere den Pfad
                query = parsed.query
                if 'd=' in query:
                    path = query.split('d=')[-1].split('&')[0]
                    path = unquote(path)
                    if os.path.exists(path):
                        return path

            # Versuche Bild herunterzuladen
            resp = requests.get(url, timeout=10)
            if resp.ok:
                # Temporäre Datei erstellen
                suffix = os.path.splitext(parsed.path)[1] or '.jpg'
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(resp.content)
                    return f.name
        except Exception as e:
            logger.warning(f"Could not fetch image from URL {url}: {e}")

    # Label Studio local-files Format (relativer Pfad)
    if '/data/local-files/' in url:
        if '?d=' in url or 'd=' in url:
            path = url.split('d=')[-1].split('&')[0]
            path = unquote(path)
            if os.path.exists(path):
                return path

    # Label Studio upload Format
    if '/data/upload/' in url:
        parts = url.split('/data/upload/')[-1]
        possible_paths = [
            f"/home/jonas/.local/share/label-studio/media/upload/{parts}",
            f"/var/lib/label-studio/media/upload/{parts}",
            os.path.expanduser(f"~/.local/share/label-studio/media/upload/{parts}"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p

    # Versuche verschiedene Basis-Pfade
    project_root = Path(__file__).parent.parent.parent
    possible_bases = [
        project_root / "data",
        Path("/home/jonas/.local/share/label-studio/media"),
        Path.home() / ".local/share/label-studio/media",
    ]

    for base in possible_bases:
        possible = base / url.lstrip('/')
        if possible.exists():
            return str(possible)

    return None


def _convert_to_label_studio(prediction, class_names: Dict[int, str]) -> List[Dict]:
    """
    Konvertiert YOLO Prediction zu Label Studio Annotation Format.

    Args:
        prediction: YOLO Results Objekt
        class_names: Mapping von Class-ID zu Namen

    Returns:
        Liste von Label Studio Annotation-Objekten
    """
    annotations = []

    if prediction.boxes is None:
        return annotations

    # Originalbild-Größe
    orig_shape = prediction.orig_shape  # (height, width)
    img_height, img_width = orig_shape

    for i, box in enumerate(prediction.boxes):
        # Koordinaten (xyxy Format)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Zu Prozent konvertieren (Label Studio Format)
        x_percent = (x1 / img_width) * 100
        y_percent = (y1 / img_height) * 100
        width_percent = ((x2 - x1) / img_width) * 100
        height_percent = ((y2 - y1) / img_height) * 100

        # Class und Confidence
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, f"class_{class_id}")
        confidence = float(box.conf[0])

        annotation = {
            "id": f"yolo_{i}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": img_width,
            "original_height": img_height,
            "value": {
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rotation": 0,
                "rectanglelabels": [class_name]
            },
            "score": confidence
        }
        annotations.append(annotation)

    return annotations


def _calculate_avg_score(prediction) -> float:
    """Berechnet den durchschnittlichen Confidence-Score."""
    if prediction.boxes is None or len(prediction.boxes) == 0:
        return 0.0

    scores = [float(box.conf[0]) for box in prediction.boxes]
    return sum(scores) / len(scores)


def run_ml_backend(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 9090,
    confidence: float = 0.25,
    debug: bool = False
):
    """
    Startet den ML Backend Server.

    Args:
        model_path: Pfad zum YOLO Modell (.pt)
        host: Host-Adresse
        port: Port-Nummer
        confidence: Confidence-Threshold
        debug: Debug-Modus
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    app = create_app(model_path, confidence)

    logger.info(f"Starting ML Backend on {host}:{port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Confidence threshold: {confidence}")
    logger.info(f"Connect in Label Studio: http://localhost:{port}")

    # Verwende waitress statt werkzeug wenn verfügbar (robuster bei umgeleiteter stdout)
    try:
        from waitress import serve
        logger.info("Using waitress server")
        serve(app, host=host, port=port)
    except ImportError:
        # Fallback zu Flask dev server mit expliziten Optionen
        logger.info("Using Flask development server")
        from werkzeug.serving import make_server
        server = make_server(host, port, app, threaded=True)
        server.serve_forever()


# CLI Entry Point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO ML Backend for Label Studio")
    parser.add_argument("--model", "-m", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", "-p", type=int, default=9090, help="Port number")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    run_ml_backend(
        model_path=args.model,
        host=args.host,
        port=args.port,
        confidence=args.confidence,
        debug=args.debug
    )
