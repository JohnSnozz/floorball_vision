"""Video Utilities for Floorball Vision.

Common video operations used across the project.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


def get_video_metadata(video_path: str) -> dict:
    """
    Extrahiert Video-Metadaten.

    Args:
        video_path: Pfad zur Videodatei

    Returns:
        dict: Metadaten (width, height, fps, duration, codec, frame_count)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Konnte Video nicht öffnen: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frame_count": frame_count,
            "duration": frame_count / fps if fps > 0 else 0,
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
    finally:
        cap.release()


def generate_thumbnail(video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
    """
    Generiert ein Thumbnail aus dem Video.

    Args:
        video_path: Pfad zur Videodatei
        output_path: Pfad für das Thumbnail
        timestamp: Zeitpunkt in Sekunden (default: 1.0)

    Returns:
        bool: True wenn erfolgreich
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Konnte Video nicht öffnen: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            # Falls Frame nicht verfügbar, erstes Frame nehmen
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if ret:
            # Thumbnail auf max 320px Breite skalieren
            height, width = frame.shape[:2]
            if width > 320:
                scale = 320 / width
                new_width = 320
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Speichern
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, frame)
            return True

        return False
    finally:
        cap.release()


def extract_frame(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """
    Extrahiert einen einzelnen Frame aus dem Video.

    Args:
        video_path: Pfad zur Videodatei
        timestamp: Zeitpunkt in Sekunden

    Returns:
        Frame als numpy array oder None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def read_video(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Read video frames as a generator.

    Args:
        video_path: Path to video file

    Yields:
        Video frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
        return info
    finally:
        cap.release()


def save_video(
    frames: list,
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v'
) -> None:
    """
    Save frames as a video file.

    Args:
        frames: List of frames (numpy arrays)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec (e.g., 'mp4v', 'XVID')
    """
    if not frames:
        raise ValueError("No frames to save")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class VideoWriter:
    """Context manager for writing video frames."""

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v'
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.size = size
        self.codec = codec
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video."""
        if self.writer is None:
            if self.size is None:
                self.size = (frame.shape[1], frame.shape[0])

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, self.size
            )

        self.writer.write(frame)
