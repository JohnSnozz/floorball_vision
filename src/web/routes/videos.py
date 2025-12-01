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


@videos_bp.route("/<video_id>/thumbnail", methods=["GET"])
def thumbnail(video_id):
    """Thumbnail für ein Video abrufen."""
    from flask import send_file, abort

    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    if not video.file_path:
        abort(404)

    # Thumbnail-Pfad: video_name_thumb.jpg
    base_path = video.file_path.rsplit(".", 1)[0]
    thumbnail_path = base_path + "_thumb.jpg"

    if not os.path.exists(thumbnail_path):
        abort(404)

    return send_file(thumbnail_path, mimetype="image/jpeg")


@videos_bp.route("/<video_id>/stream", methods=["GET"])
def stream_video(video_id):
    """Video streamen mit Range-Request Support."""
    from flask import send_file, abort, Response, request
    from pathlib import Path

    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    if not video.file_path:
        abort(404)

    video_path = Path(video.file_path)
    if not video_path.is_absolute():
        # Relativer Pfad - vom Projekt-Root aus
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        video_path = project_root / video.file_path

    if not video_path.exists():
        abort(404)

    file_size = video_path.stat().st_size

    # Bestimme MIME-Type
    ext = video_path.suffix.lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
    }
    mimetype = mime_types.get(ext, 'video/mp4')

    # Range-Request Support für Seeking
    range_header = request.headers.get('Range')

    if range_header:
        # Parse Range header
        byte_start = 0
        byte_end = file_size - 1

        if range_header.startswith('bytes='):
            range_spec = range_header[6:]
            if '-' in range_spec:
                start_str, end_str = range_spec.split('-', 1)
                if start_str:
                    byte_start = int(start_str)
                if end_str:
                    byte_end = int(end_str)

        # Chunk-Größe begrenzen (1MB)
        chunk_size = 1024 * 1024
        byte_end = min(byte_end, byte_start + chunk_size - 1, file_size - 1)

        content_length = byte_end - byte_start + 1

        def generate():
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        response = Response(
            generate(),
            status=206,
            mimetype=mimetype,
            direct_passthrough=True
        )
        response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = content_length
        return response

    # Kein Range-Request - ganzes Video senden
    return send_file(video_path, mimetype=mimetype)
