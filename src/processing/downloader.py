"""
YouTube Video Downloader

Verwendet yt-dlp für Video-Downloads.
"""
import os
from typing import Callable, Optional
import yt_dlp


class DownloadError(Exception):
    """Fehler beim Video-Download."""
    pass


def get_video_info(url: str) -> dict:
    """
    Video-Informationen abrufen ohne Download.

    Args:
        url: YouTube URL

    Returns:
        dict: Video-Metadaten (title, duration, etc.)
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title"),
                "duration": info.get("duration"),
                "description": info.get("description"),
                "thumbnail": info.get("thumbnail"),
                "uploader": info.get("uploader"),
                "view_count": info.get("view_count"),
                "upload_date": info.get("upload_date"),
            }
    except Exception as e:
        raise DownloadError(f"Konnte Video-Info nicht abrufen: {str(e)}")


def download_video(
    url: str,
    output_dir: str,
    video_id: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    max_resolution: int = 1080,
) -> dict:
    """
    Video von YouTube herunterladen.

    Args:
        url: YouTube URL
        output_dir: Zielverzeichnis
        video_id: ID für Dateinamen
        progress_callback: Callback für Fortschritt (0-100)
        max_resolution: Maximale Auflösung

    Returns:
        dict: Download-Ergebnis mit Dateiinfo
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ausgabe-Template
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    # Progress Hook
    def progress_hook(d):
        if d["status"] == "downloading" and progress_callback:
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                progress = (downloaded / total) * 100
                progress_callback(progress)

    ydl_opts = {
        "format": f"bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook],
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Finde die heruntergeladene Datei
            filename = f"{video_id}.mp4"
            file_path = os.path.join(output_dir, filename)

            # Falls mit anderer Extension gespeichert
            if not os.path.exists(file_path):
                for ext in ["mp4", "webm", "mkv"]:
                    alt_path = os.path.join(output_dir, f"{video_id}.{ext}")
                    if os.path.exists(alt_path):
                        file_path = alt_path
                        filename = f"{video_id}.{ext}"
                        break

            if not os.path.exists(file_path):
                raise DownloadError("Download abgeschlossen, aber Datei nicht gefunden")

            return {
                "filename": filename,
                "file_path": file_path,
                "title": info.get("title", "Unbekannt"),
                "duration": info.get("duration"),
                "thumbnail": info.get("thumbnail"),
            }

    except yt_dlp.DownloadError as e:
        raise DownloadError(f"Download fehlgeschlagen: {str(e)}")
    except Exception as e:
        raise DownloadError(f"Unbekannter Fehler: {str(e)}")
