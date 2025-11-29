"""
Video API Routes

Endpoints für Video-Upload, YouTube-Download und Video-Verwaltung.
"""
import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from src.web.extensions import db
from src.db.models import Video

videos_bp = Blueprint("videos", __name__)


def allowed_file(filename):
    """Prüft ob die Dateiendung erlaubt ist."""
    allowed = current_app.config.get("ALLOWED_VIDEO_EXTENSIONS", {"mp4", "avi", "mov", "mkv", "webm"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def video_to_dict(video: Video) -> dict:
    """Konvertiert Video-Model zu Dictionary."""
    return {
        "id": str(video.id),
        "filename": video.filename,
        "original_filename": video.original_filename,
        "duration": video.duration,
        "fps": video.fps,
        "width": video.width,
        "height": video.height,
        "source_type": video.source_type,
        "source_url": video.source_url,
        "status": video.status,
        "error_message": video.error_message,
        "created_at": video.created_at.isoformat() if video.created_at else None,
    }


@videos_bp.route("", methods=["GET"])
def list_videos():
    """Liste aller Videos."""
    videos = db.session.query(Video).order_by(Video.created_at.desc()).all()
    return jsonify([video_to_dict(v) for v in videos])


@videos_bp.route("/<video_id>", methods=["GET"])
def get_video(video_id):
    """Video Details abrufen."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404
    return jsonify(video_to_dict(video))


@videos_bp.route("/<video_id>/status", methods=["GET"])
def get_video_status(video_id):
    """Video Status abrufen (für Polling während Download)."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404
    return jsonify({
        "id": str(video.id),
        "status": video.status,
        "error_message": video.error_message,
    })


@videos_bp.route("/upload", methods=["POST"])
def upload_video():
    """Video-Datei hochladen."""
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei im Request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Keine Datei ausgewählt"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Dateityp nicht erlaubt"}), 400

    # Eindeutigen Dateinamen generieren
    original_filename = secure_filename(file.filename)
    video_id = uuid.uuid4()
    ext = original_filename.rsplit(".", 1)[1].lower()
    filename = f"{video_id}.{ext}"

    # Speicherpfad
    videos_path = current_app.config.get("VIDEOS_PATH", "data/videos")
    os.makedirs(videos_path, exist_ok=True)
    file_path = os.path.join(videos_path, filename)

    # Datei speichern
    file.save(file_path)

    # Video in DB erstellen
    video = Video(
        id=video_id,
        filename=filename,
        original_filename=original_filename,
        file_path=file_path,
        source_type="upload",
        status="processing",
    )
    db.session.add(video)
    db.session.commit()

    # Celery Task für Metadaten-Extraktion starten
    from src.processing.tasks import process_uploaded_video
    process_uploaded_video.delay(str(video_id))

    return jsonify(video_to_dict(video)), 201


@videos_bp.route("/youtube", methods=["POST"])
def download_youtube():
    """YouTube Video herunterladen."""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL fehlt"}), 400

    url = data["url"]

    # Validiere YouTube URL (einfache Prüfung)
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Ungültige YouTube URL"}), 400

    # Video in DB erstellen
    video_id = uuid.uuid4()
    video = Video(
        id=video_id,
        filename="",  # Wird nach Download gesetzt
        original_filename="",
        file_path="",
        source_type="youtube",
        source_url=url,
        status="downloading",
    )
    db.session.add(video)
    db.session.commit()

    # Celery Task starten
    from src.processing.tasks import download_youtube_video
    download_youtube_video.delay(str(video_id), url)

    return jsonify(video_to_dict(video)), 202


@videos_bp.route("/<video_id>", methods=["DELETE"])
def delete_video(video_id):
    """Video löschen."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Datei löschen falls vorhanden
    if video.file_path and os.path.exists(video.file_path):
        try:
            os.remove(video.file_path)
        except OSError:
            pass  # Datei konnte nicht gelöscht werden

    # Thumbnail löschen falls vorhanden
    thumbnail_path = video.file_path.rsplit(".", 1)[0] + "_thumb.jpg" if video.file_path else None
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            os.remove(thumbnail_path)
        except OSError:
            pass

    # Aus DB löschen
    db.session.delete(video)
    db.session.commit()

    return jsonify({"message": "Video gelöscht"}), 200
