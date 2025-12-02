"""
Team Assigner - CLIP-basierte Team-Zuweisung für Spieler.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class TeamAssigner:
    """
    Weist Spielern Teams zu basierend auf CLIP-Analyse der Trikotfarben.

    Verwendet das Fashion-CLIP Modell für bessere Farberkennung bei Kleidung.
    """

    # Singleton-Instanzen für CLIP-Modell (teuer zu laden)
    _model = None
    _processor = None

    def __init__(
        self,
        team_config: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialisiert den Team-Assigner.

        Args:
            team_config: Konfiguration der Teams mit Farbdeskriptoren
                {
                    "team1": {"color": "blue shirt", "displayColor": "#3B82F6"},
                    "team2": {"color": "white shirt", "displayColor": "#FFFFFF"},
                    "referee": {"color": "pink shirt", "displayColor": "#EC4899"}
                }
        """
        self.team_config = team_config or {
            "team1": {"color": "blue shirt", "displayColor": "#3B82F6"},
            "team2": {"color": "white shirt", "displayColor": "#FFFFFF"},
            "referee": {"color": "pink shirt", "displayColor": "#EC4899"}
        }

        # CLIP-Modell lazy laden
        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls):
        """Lädt das CLIP-Modell (Singleton)."""
        if cls._model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                print("Loading CLIP model for team assignment...")
                cls._model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
                cls._processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
                print("CLIP model loaded successfully")
            except Exception as e:
                print(f"ERROR loading CLIP model: {e}")
                cls._model = None
                cls._processor = None

    @property
    def is_available(self) -> bool:
        """Prüft ob das CLIP-Modell verfügbar ist."""
        return self._model is not None and self._processor is not None

    def assign_team(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> Tuple[str, float]:
        """
        Weist einer Detection ein Team zu.

        Args:
            frame: Video-Frame (BGR)
            bbox: Bounding Box [x1, y1, x2, y2]

        Returns:
            Tuple (team_name, confidence)
        """
        if not self.is_available:
            return "unknown", 0.0

        from PIL import Image
        import torch

        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Spieler-Crop
        player_img = frame[y1:y2, x1:x2]
        if player_img.size == 0:
            return "unknown", 0.0

        # BGR -> RGB -> PIL
        rgb_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Team-Deskriptoren
        classes = [
            self.team_config["team1"]["color"],
            self.team_config["team2"]["color"],
            self.team_config["referee"]["color"]
        ]

        # CLIP Inference
        inputs = self._processor(
            text=classes,
            images=pil_img,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0].tolist()

        # Beste Übereinstimmung
        best_idx = probs.index(max(probs))
        best_prob = probs[best_idx]

        if best_idx == 0:
            return "team1", best_prob
        elif best_idx == 1:
            return "team2", best_prob
        else:
            return "referee", best_prob

    def get_display_color(self, team: str) -> str:
        """Gibt die Display-Farbe für ein Team zurück."""
        if team in self.team_config:
            return self.team_config[team].get("displayColor", "#888888")
        return "#888888"
