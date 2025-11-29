"""
Labeling Sync

Synchronisiert Frames zwischen Video und Label Studio.
"""
import os
import random
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .client import LabelStudioClient, LabelStudioError


class LabelingSync:
    """Synchronisiert Frames mit Label Studio."""

    def __init__(
        self,
        client: Optional[LabelStudioClient] = None,
        frames_dir: Optional[str] = None
    ):
        """
        Initialisiert den Sync.

        Args:
            client: Label Studio Client
            frames_dir: Verzeichnis für extrahierte Frames
        """
        self.client = client or LabelStudioClient()
        self.frames_dir = Path(frames_dir or "data/labeling/frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def extract_random_frames(
        self,
        video_path: str,
        num_frames: int = 100,
        batch_id: Optional[str] = None
    ) -> dict:
        """
        Extrahiert zufällige Frames aus einem Video.

        Args:
            video_path: Pfad zum Video
            num_frames: Anzahl zu extrahierender Frames
            batch_id: ID für diesen Batch (default: generiert)

        Returns:
            dict mit batch_id und Liste der Frame-Pfade
        """
        from src.utils.video_utils import get_video_metadata

        video_path = Path(video_path)
        if not video_path.exists():
            raise LabelStudioError(f"Video nicht gefunden: {video_path}")

        # Batch-ID generieren
        batch_id = batch_id or str(uuid4())[:8]
        batch_dir = self.frames_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Video-Metadaten holen
        metadata = get_video_metadata(str(video_path))
        duration = metadata.get("duration", 0)
        fps = metadata.get("fps", 25)

        if duration <= 0:
            raise LabelStudioError("Konnte Video-Dauer nicht ermitteln")

        total_frames = int(duration * fps)

        # Zufällige Frame-Indizes wählen
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = sorted(random.sample(range(total_frames), num_frames))

        # Frames extrahieren
        extracted = self._extract_frames_ffmpeg(
            video_path=str(video_path),
            frame_indices=frame_indices,
            output_dir=batch_dir,
            fps=fps
        )

        return {
            "batch_id": batch_id,
            "batch_dir": str(batch_dir),
            "frames": extracted,
            "total_extracted": len(extracted),
            "video_path": str(video_path),
        }

    def _extract_frames_ffmpeg(
        self,
        video_path: str,
        frame_indices: list,
        output_dir: Path,
        fps: float
    ) -> list:
        """
        Extrahiert Frames mit ffmpeg.

        Args:
            video_path: Pfad zum Video
            frame_indices: Liste der Frame-Indizes
            output_dir: Ausgabe-Verzeichnis
            fps: Frames per Second

        Returns:
            Liste der extrahierten Frame-Pfade
        """
        import subprocess

        extracted = []

        for idx in frame_indices:
            # Zeit in Sekunden berechnen
            timestamp = idx / fps
            output_file = output_dir / f"frame_{idx:06d}.jpg"

            # ffmpeg Befehl
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",  # Hohe Qualität
                "-y",  # Überschreiben
                str(output_file)
            ]

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                    timeout=30
                )
                if output_file.exists():
                    extracted.append(str(output_file))
            except subprocess.CalledProcessError:
                continue
            except subprocess.TimeoutExpired:
                continue

        return extracted

    def upload_batch_to_project(
        self,
        batch_id: str,
        project_id: int
    ) -> dict:
        """
        Lädt einen Batch von Frames zu Label Studio hoch.

        Args:
            batch_id: Batch-ID
            project_id: Label Studio Projekt-ID

        Returns:
            Upload-Ergebnis mit Task-IDs
        """
        batch_dir = self.frames_dir / batch_id

        if not batch_dir.exists():
            raise LabelStudioError(f"Batch nicht gefunden: {batch_id}")

        # Alle Bilder im Batch finden
        image_paths = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.png"))

        if not image_paths:
            raise LabelStudioError(f"Keine Bilder in Batch {batch_id}")

        # Task-IDs vor dem Upload holen (um neue zu identifizieren)
        existing_tasks = self.client.list_tasks(project_id)
        existing_ids = {t["id"] for t in existing_tasks}

        # Hochladen
        uploaded = 0
        failed = 0
        errors = []

        for path in image_paths:
            try:
                self.client.upload_image(project_id, str(path))
                uploaded += 1
            except LabelStudioError as e:
                failed += 1
                errors.append(str(e))

        # Neue Task-IDs ermitteln
        new_tasks = self.client.list_tasks(project_id)
        new_task_ids = [t["id"] for t in new_tasks if t["id"] not in existing_ids]

        return {
            "batch_id": batch_id,
            "project_id": project_id,
            "uploaded": uploaded,
            "failed": failed,
            "errors": errors,
            "task_ids": new_task_ids,
            "first_task_id": min(new_task_ids) if new_task_ids else None,
        }

    def create_project_with_batch(
        self,
        video_id: str,
        video_path: str,
        num_frames: int = 100,
        project_title: Optional[str] = None
    ) -> dict:
        """
        Erstellt ein neues Projekt und lädt Frames hoch.

        Convenience-Methode die:
        1. Frames aus Video extrahiert
        2. Label Studio Projekt erstellt
        3. Frames hochlädt

        Args:
            video_id: Video-ID aus der Datenbank
            video_path: Pfad zum Video
            num_frames: Anzahl Frames
            project_title: Projekt-Titel (default: generiert)

        Returns:
            dict mit Projekt- und Batch-Infos
        """
        # 1. Frames extrahieren
        batch_result = self.extract_random_frames(
            video_path=video_path,
            num_frames=num_frames,
            batch_id=f"v{video_id[:8]}"
        )

        # 2. Projekt erstellen
        if not project_title:
            project_title = f"Floorball - Video {video_id[:8]}"

        project = self.client.create_project(
            title=project_title,
            description=f"Frames aus Video {video_id}"
        )

        project_id = project["id"]

        # 3. Frames hochladen
        upload_result = self.upload_batch_to_project(
            batch_id=batch_result["batch_id"],
            project_id=project_id
        )

        return {
            "project_id": project_id,
            "project_url": self.client.get_labeling_url(project_id),
            "batch_id": batch_result["batch_id"],
            "frames_extracted": batch_result["total_extracted"],
            "frames_uploaded": upload_result["uploaded"],
        }

    def get_batch_status(self, batch_id: str) -> dict:
        """
        Gibt den Status eines Batches zurück.

        Args:
            batch_id: Batch-ID

        Returns:
            dict mit Batch-Informationen
        """
        batch_dir = self.frames_dir / batch_id

        if not batch_dir.exists():
            return {"exists": False, "batch_id": batch_id}

        frames = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.png"))

        return {
            "exists": True,
            "batch_id": batch_id,
            "batch_dir": str(batch_dir),
            "frame_count": len(frames),
            "frames": [str(f) for f in frames],
        }

    def list_batches(self) -> list:
        """Listet alle Batches."""
        batches = []
        for batch_dir in self.frames_dir.iterdir():
            if batch_dir.is_dir():
                batches.append(self.get_batch_status(batch_dir.name))
        return batches

    def delete_batch(self, batch_id: str) -> bool:
        """
        Löscht einen Batch.

        Args:
            batch_id: Batch-ID

        Returns:
            True wenn erfolgreich
        """
        import shutil
        batch_dir = self.frames_dir / batch_id

        if batch_dir.exists():
            shutil.rmtree(batch_dir)
            return True
        return False
