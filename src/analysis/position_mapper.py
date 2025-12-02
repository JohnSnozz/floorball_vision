"""
Position Mapper - Pixel zu Feldkoordinaten Transformation.

Transformiert verzerrte Pixel-Koordinaten zu echten Spielfeld-Koordinaten
unter Berücksichtigung von Fisheye-Entzerrung und Homography.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CalibrationParams:
    """Parameter für die Koordinaten-Transformation."""
    homography_matrix: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray  # [k1, k2, k3, k4]
    zoom_out: float = 1.0
    rotation: float = 0.0
    video_width: int = 1920
    video_height: int = 1080


class PositionMapper:
    """
    Transformiert Pixel-Koordinaten zu Spielfeld-Koordinaten.

    Berücksichtigt:
    - Fisheye-Entzerrung (k1-k4 Distortion)
    - Zoom-Korrektur
    - Rotation
    - Homography (Perspektiv-Transformation)
    """

    # Floorball-Spielfeld Dimensionen
    FIELD_LENGTH = 40.0  # Meter
    FIELD_WIDTH = 20.0   # Meter

    def __init__(self, params: CalibrationParams):
        """
        Initialisiert den Position Mapper.

        Args:
            params: Kalibrierungs-Parameter
        """
        self.params = params

    @classmethod
    def from_calibration_data(
        cls,
        cal_data: Dict[str, Any],
        video_width: int = 1920,
        video_height: int = 1080,
        lens_profile_loader=None
    ) -> "PositionMapper":
        """
        Erstellt einen PositionMapper aus Kalibrierungsdaten.

        Args:
            cal_data: Kalibrierungs-Dictionary aus der Datenbank
            video_width: Video-Breite in Pixel
            video_height: Video-Höhe in Pixel
            lens_profile_loader: Funktion zum Laden von Lens-Profilen

        Returns:
            PositionMapper-Instanz
        """
        # Homography-Matrix extrahieren
        homography_matrix = None
        if cal_data.get("homography_matrix"):
            homography_matrix = np.array(cal_data["homography_matrix"], dtype=np.float32)
        elif cal_data.get("combined_calibration", {}).get("homography_matrix"):
            homography_matrix = np.array(
                cal_data["combined_calibration"]["homography_matrix"],
                dtype=np.float32
            )

        if homography_matrix is None:
            raise ValueError("Keine Homography-Matrix in Kalibrierung gefunden")

        # Undistort-Parameter
        undistort_params = cal_data.get("undistort_params", {})
        k1 = undistort_params.get("k1", 0.0)
        k2 = undistort_params.get("k2", 0.0)
        k3 = undistort_params.get("k3", 0.0)
        k4 = undistort_params.get("k4", 0.0)
        zoom_out = undistort_params.get("zoom_out", 1.0)
        rotation = undistort_params.get("rotation", 0.0)
        lens_profile_id = undistort_params.get("lens_profile_id") or cal_data.get("lens_profile_id")

        # Kamera-Matrix aus Lens-Profil oder Default
        camera_matrix = None
        dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float64)

        # Bei Custom-Modus: base_profile_id verwenden
        base_profile_id = undistort_params.get("base_profile_id")
        profile_to_load = base_profile_id if lens_profile_id == "custom" else lens_profile_id

        if profile_to_load and lens_profile_loader:
            profile = lens_profile_loader(profile_to_load)
            if profile and profile.get("fisheye_params"):
                camera_matrix = np.array(
                    profile["fisheye_params"]["camera_matrix"],
                    dtype=np.float64
                )
                profile_resolution = profile.get("resolution", {})

                # k1-k4 aus Profil wenn nicht custom
                if lens_profile_id != "custom" and k1 == 0 and k2 == 0:
                    coeffs = profile["fisheye_params"].get("distortion_coeffs", [0, 0, 0, 0])
                    dist_coeffs = np.array(coeffs[:4], dtype=np.float64)

                # Kamera-Matrix skalieren falls nötig
                profile_w = profile_resolution.get("width", video_width)
                profile_h = profile_resolution.get("height", video_height)
                if profile_w != video_width or profile_h != video_height:
                    camera_matrix = cls._scale_camera_matrix(
                        camera_matrix,
                        (profile_w, profile_h),
                        (video_width, video_height)
                    )

        # Fallback: Standard-Kamera-Matrix
        if camera_matrix is None:
            focal = video_width
            cx, cy = video_width / 2, video_height / 2
            camera_matrix = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ], dtype=np.float64)

        params = CalibrationParams(
            homography_matrix=homography_matrix,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            zoom_out=zoom_out,
            rotation=rotation,
            video_width=video_width,
            video_height=video_height
        )

        return cls(params)

    @staticmethod
    def _scale_camera_matrix(
        camera_matrix: np.ndarray,
        from_size: Tuple[int, int],
        to_size: Tuple[int, int]
    ) -> np.ndarray:
        """Skaliert Kamera-Matrix auf neue Auflösung."""
        scale_x = to_size[0] / from_size[0]
        scale_y = to_size[1] / from_size[1]

        scaled = camera_matrix.copy()
        scaled[0, 0] *= scale_x  # fx
        scaled[1, 1] *= scale_y  # fy
        scaled[0, 2] *= scale_x  # cx
        scaled[1, 2] *= scale_y  # cy

        return scaled

    def pixel_to_field(self, px: float, py: float) -> Tuple[float, float]:
        """
        Transformiert verzerrte Pixel-Koordinaten zu Feld-Koordinaten.

        Der Prozess:
        1. Fisheye-Entzerrung mit angepasster Kamera-Matrix (zoom_out)
        2. Rotation rückgängig machen
        3. Homography anwenden

        Args:
            px: Pixel X-Koordinate
            py: Pixel Y-Koordinate

        Returns:
            Tuple (field_x, field_y) in Metern
        """
        w = self.params.video_width
        h = self.params.video_height
        cx_img, cy_img = w / 2, h / 2

        undist_px, undist_py = px, py

        # 1. Fisheye-Entzerrung
        k1, k2, k3, k4 = self.params.dist_coeffs
        if k1 != 0 or k2 != 0 or k3 != 0 or k4 != 0:
            dist_coeffs_arr = self.params.dist_coeffs.reshape(4, 1)

            # Angepasste Kamera-Matrix für Output (mit zoom_out)
            new_K = self.params.camera_matrix.copy()
            if self.params.zoom_out != 1.0:
                new_K[0, 0] /= self.params.zoom_out
                new_K[1, 1] /= self.params.zoom_out

            pts = np.array([[[px, py]]], dtype=np.float64)

            undistorted = cv2.fisheye.undistortPoints(
                pts,
                self.params.camera_matrix,
                dist_coeffs_arr,
                R=np.eye(3),
                P=new_K
            )
            undist_px, undist_py = undistorted[0, 0]

        # 2. Rotation rückgängig machen (NEGATIVER Winkel!)
        if abs(self.params.rotation) > 0.01:
            angle_rad = np.radians(-self.params.rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            centered_x = undist_px - cx_img
            centered_y = undist_py - cy_img
            rot_x = centered_x * cos_a - centered_y * sin_a
            rot_y = centered_x * sin_a + centered_y * cos_a
            undist_px = rot_x + cx_img
            undist_py = rot_y + cy_img

        # 3. Homography anwenden
        pts = np.array([[[undist_px, undist_py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.params.homography_matrix)
        field_x, field_y = transformed[0, 0]

        return float(field_x), float(field_y)

    def is_valid_field_position(
        self,
        field_x: float,
        field_y: float,
        tolerance: float = 5.0
    ) -> bool:
        """
        Prüft ob eine Feldposition plausibel ist.

        Args:
            field_x: X-Koordinate in Metern
            field_y: Y-Koordinate in Metern
            tolerance: Toleranz ausserhalb des Feldes

        Returns:
            True wenn Position plausibel
        """
        return (
            -tolerance <= field_x <= self.FIELD_LENGTH + tolerance and
            -tolerance <= field_y <= self.FIELD_WIDTH + tolerance
        )

    def transform_tracking_data(
        self,
        tracking_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transformiert alle Tracking-Daten zu Feldkoordinaten.

        Args:
            tracking_data: Tracking-Daten mit frames und detections

        Returns:
            Positions-Daten mit Feldkoordinaten
        """
        track_team_assignments = tracking_data.get("track_team_assignments", {})

        positions_data = {
            "field_dimensions": {
                "length": self.FIELD_LENGTH,
                "width": self.FIELD_WIDTH
            },
            "frames": [],
            "debug_stats": {
                "total_detections": 0,
                "skipped_not_in_field": 0,
                "skipped_transform_error": 0,
                "skipped_out_of_bounds": 0,
                "valid_positions": 0
            }
        }

        stats = positions_data["debug_stats"]

        for frame in tracking_data.get("frames", []):
            frame_positions = {
                "frame_idx": frame["frame_idx"],
                "timestamp": frame["timestamp"],
                "players": [],
                "referees": [],
                "goalkeepers": [],
                "unknown": []
            }

            for det in frame.get("detections", []):
                stats["total_detections"] += 1

                if not det.get("in_field", True):
                    stats["skipped_not_in_field"] += 1
                    continue

                # Fuss-Position
                bbox = det["bbox"]
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]

                # Transformation
                try:
                    field_x, field_y = self.pixel_to_field(foot_x, foot_y)
                except Exception as e:
                    stats["skipped_transform_error"] += 1
                    continue

                # Validierung
                if not self.is_valid_field_position(field_x, field_y):
                    stats["skipped_out_of_bounds"] += 1
                    continue

                stats["valid_positions"] += 1

                track_id = det.get("track_id")

                # Team aus finalen Assignments
                team = "unknown"
                if track_id is not None:
                    track_id_str = str(track_id)
                    if track_id_str in track_team_assignments:
                        team = track_team_assignments[track_id_str].get("team", "unknown")
                    elif det.get("team"):
                        team = det["team"]

                position = {
                    "track_id": track_id,
                    "x": round(field_x, 2),
                    "y": round(field_y, 2),
                    "team": team,
                    "confidence": det.get("conf", 0),
                    "jersey_number": det.get("jersey_number")
                }

                # Kategorisierung
                class_name = det.get("class_name", "").lower()
                if "goalkeeper" in class_name or "goalie" in class_name:
                    frame_positions["goalkeepers"].append(position)
                elif team == "referee":
                    frame_positions["referees"].append(position)
                elif team in ["team1", "team2"]:
                    frame_positions["players"].append(position)
                else:
                    frame_positions["unknown"].append(position)

            positions_data["frames"].append(frame_positions)

        return positions_data
