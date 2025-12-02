"""
Jersey OCR - Rückennummer-Erkennung mit EasyOCR.
"""
import cv2
import numpy as np
import re
from typing import Tuple, Optional, List


class JerseyOCR:
    """
    Erkennt Rückennummern auf Spieler-Crops mit EasyOCR.
    """

    # Singleton-Instanz für OCR Reader (teuer zu laden)
    _reader = None

    def __init__(self, use_gpu: bool = False):
        """
        Initialisiert den Jersey-OCR Reader.

        Args:
            use_gpu: GPU für OCR verwenden (kann Probleme verursachen)
        """
        self.use_gpu = use_gpu
        self._ensure_reader_loaded()

    @classmethod
    def _ensure_reader_loaded(cls, use_gpu: bool = False):
        """Lädt den EasyOCR Reader (Singleton)."""
        if cls._reader is None:
            try:
                import easyocr
                print("Loading EasyOCR for jersey number recognition...")
                cls._reader = easyocr.Reader(['en'], gpu=use_gpu)
                print("EasyOCR loaded successfully")
            except Exception as e:
                print(f"ERROR loading EasyOCR: {e}")
                cls._reader = None

    @property
    def is_available(self) -> bool:
        """Prüft ob der OCR Reader verfügbar ist."""
        return self._reader is not None

    def detect(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> Tuple[Optional[int], float]:
        """
        Erkennt Rückennummer auf einem Spieler-Crop.

        Args:
            frame: Video-Frame (BGR)
            bbox: Bounding Box [x1, y1, x2, y2]

        Returns:
            Tuple (jersey_number oder None, confidence)
        """
        if not self.is_available:
            return None, 0.0

        x1, y1, x2, y2 = [int(c) for c in bbox]
        h = y2 - y1
        w = x2 - x1

        # Nur den oberen Teil (Rücken/Brust) des Spielers betrachten
        # Typischerweise ist die Nummer im oberen 60% der Box
        crop_y1 = y1
        crop_y2 = y1 + int(h * 0.6)
        crop_x1 = x1 + int(w * 0.1)  # Etwas Rand abschneiden
        crop_x2 = x2 - int(w * 0.1)

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None, 0.0

        player_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if player_crop.size == 0:
            return None, 0.0

        try:
            # OCR durchführen - nur Ziffern erlauben
            results = self._reader.readtext(player_crop, allowlist='0123456789')

            # Nur Zahlen 1-99 akzeptieren
            for (bbox_ocr, text, conf) in results:
                numbers = re.findall(r'\d+', text)
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if 1 <= num <= 99:
                            return num, float(conf)
                    except ValueError:
                        continue

            return None, 0.0

        except Exception as e:
            return None, 0.0
