"""
Celery Tasks

Background Tasks für Video-Verarbeitung.
"""
import os

from src.web.extensions import db, celery
from src.db.models import Video


def get_flask_app():
    """Flask App für Celery Context erstellen."""
    from src.web.app import create_app
    return create_app()


@celery.task(bind=True)
def download_youtube_video(self, video_id: str, url: str):
    """
    YouTube Video herunterladen.

    Args:
        video_id: UUID des Videos in der DB
        url: YouTube URL
    """
    from src.processing.downloader import download_video, DownloadError
    from src.web.config import BaseConfig

    app = get_flask_app()
    with app.app_context():
        video = db.session.get(Video, video_id)
        if not video:
            return {"error": "Video nicht gefunden"}

        try:
            # Download-Pfad
            videos_path = BaseConfig.VIDEOS_PATH
            os.makedirs(videos_path, exist_ok=True)

            # Video herunterladen
            result = download_video(
                url=url,
                output_dir=videos_path,
                video_id=video_id,
                progress_callback=lambda p: self.update_state(
                    state="PROGRESS",
                    meta={"progress": p}
                )
            )

            # Video in DB aktualisieren
            video.filename = result["filename"]
            video.original_filename = result["title"]
            video.file_path = result["file_path"]
            video.duration = result.get("duration")
            video.status = "processing"
            db.session.commit()

            # Metadaten extrahieren
            process_uploaded_video.delay(video_id)

            return {"status": "success", "video_id": video_id}

        except DownloadError as e:
            video.status = "error"
            video.error_message = str(e)
            db.session.commit()
            return {"error": str(e)}

        except Exception as e:
            video.status = "error"
            video.error_message = f"Unbekannter Fehler: {str(e)}"
            db.session.commit()
            raise


@celery.task
def process_uploaded_video(video_id: str):
    """
    Hochgeladenes Video verarbeiten.

    - Metadaten extrahieren
    - Thumbnail generieren

    Args:
        video_id: UUID des Videos
    """
    from src.utils.video_utils import get_video_metadata, generate_thumbnail

    app = get_flask_app()
    with app.app_context():
        video = db.session.get(Video, video_id)
        if not video:
            return {"error": "Video nicht gefunden"}

        try:
            # Metadaten extrahieren
            if video.file_path and os.path.exists(video.file_path):
                metadata = get_video_metadata(video.file_path)
                video.duration = metadata.get("duration")
                video.fps = metadata.get("fps")
                video.width = metadata.get("width")
                video.height = metadata.get("height")
                video.codec = metadata.get("codec")

                # Thumbnail generieren
                thumb_path = video.file_path.rsplit(".", 1)[0] + "_thumb.jpg"
                generate_thumbnail(video.file_path, thumb_path)

            video.status = "ready"
            db.session.commit()

            return {"status": "success", "video_id": video_id}

        except Exception as e:
            video.status = "error"
            video.error_message = f"Verarbeitung fehlgeschlagen: {str(e)}"
            db.session.commit()
            return {"error": str(e)}
