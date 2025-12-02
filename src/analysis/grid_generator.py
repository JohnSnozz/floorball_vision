"""
Grid Generator - Spielfeld-Grid für Video-Overlay.

Generiert Spielfeld-Linien die auf das verzerrte Video-Bild
projiziert werden können (Fisheye-Korrektur berücksichtigt).
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridParams:
    """Parameter für die Grid-Generierung."""
    homography_matrix: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray  # [k1, k2, k3, k4]
    zoom_out: float = 1.0
    rotation: float = 0.0
    video_width: int = 1920
    video_height: int = 1080


class GridGenerator:
    """
    Generiert Spielfeld-Linien für Video-Overlay.

    Transformiert Feldkoordinaten zu verzerrten Pixel-Koordinaten
    damit das Grid korrekt auf dem Original-Video angezeigt wird.
    """

    # Floorball-Spielfeld Dimensionen
    FIELD_LENGTH = 40.0  # Meter
    FIELD_WIDTH = 20.0   # Meter
    GOAL_LINE_DIST = 2.85  # Meter von Endlinie
    CENTER_CIRCLE_RADIUS = 3.0  # Meter
    CREASE_RADIUS = 4.0  # Meter (Torraum)

    def __init__(self, params: GridParams):
        """
        Initialisiert den Grid Generator.

        Args:
            params: Grid-Parameter
        """
        self.params = params

        # Inverse Homography berechnen
        self.inv_homography = np.linalg.inv(params.homography_matrix)

        # Gezoomte Kamera-Matrix
        self.zoomed_K = params.camera_matrix.copy()
        self.zoomed_K[0, 0] /= params.zoom_out
        self.zoomed_K[1, 1] /= params.zoom_out

    @classmethod
    def from_calibration_data(
        cls,
        cal_data: Dict[str, Any],
        video_width: int = 1920,
        video_height: int = 1080,
        lens_profile_loader=None
    ) -> "GridGenerator":
        """
        Erstellt einen GridGenerator aus Kalibrierungsdaten.

        Args:
            cal_data: Kalibrierungs-Dictionary
            video_width: Video-Breite
            video_height: Video-Höhe
            lens_profile_loader: Funktion zum Laden von Lens-Profilen

        Returns:
            GridGenerator-Instanz
        """
        # Gleiche Logik wie PositionMapper
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

        undistort_params = cal_data.get("undistort_params", {})
        k1 = undistort_params.get("k1", 0.0)
        k2 = undistort_params.get("k2", 0.0)
        k3 = undistort_params.get("k3", 0.0)
        k4 = undistort_params.get("k4", 0.0)
        zoom_out = undistort_params.get("zoom_out", 1.0)
        rotation = undistort_params.get("rotation", 0.0)
        lens_profile_id = undistort_params.get("lens_profile_id") or cal_data.get("lens_profile_id")

        camera_matrix = None
        dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float64)

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

                if lens_profile_id != "custom" and k1 == 0 and k2 == 0:
                    coeffs = profile["fisheye_params"].get("distortion_coeffs", [0, 0, 0, 0])
                    dist_coeffs = np.array(coeffs[:4], dtype=np.float64)

                profile_w = profile_resolution.get("width", video_width)
                profile_h = profile_resolution.get("height", video_height)
                if profile_w != video_width or profile_h != video_height:
                    scale_x = video_width / profile_w
                    scale_y = video_height / profile_h
                    camera_matrix[0, 0] *= scale_x
                    camera_matrix[1, 1] *= scale_y
                    camera_matrix[0, 2] *= scale_x
                    camera_matrix[1, 2] *= scale_y

        if camera_matrix is None:
            focal = video_width
            cx, cy = video_width / 2, video_height / 2
            camera_matrix = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ], dtype=np.float64)

        params = GridParams(
            homography_matrix=homography_matrix,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            zoom_out=zoom_out,
            rotation=rotation,
            video_width=video_width,
            video_height=video_height
        )

        return cls(params)

    def field_to_distorted_pixel(
        self,
        field_points: List[List[float]]
    ) -> np.ndarray:
        """
        Transformiert Feldpunkte zu verzerrten Pixel-Koordinaten.

        Workflow:
        1. Feldpunkte -> entzerrte Bildpunkte (via inverse Homographie)
        2. Rotation rückgängig machen (POSITIVER Winkel)
        3. Fisheye-Verzerrung anwenden

        Args:
            field_points: Liste von [x, y] Feldkoordinaten in Metern

        Returns:
            Array von verzerrten Pixel-Koordinaten
        """
        w = self.params.video_width
        h = self.params.video_height
        cx = self.params.camera_matrix[0, 2]
        cy = self.params.camera_matrix[1, 2]

        # 1. Feld -> Bild (entzerrt + rotiert)
        field_pts_arr = np.array([[[p[0], p[1]]] for p in field_points], dtype=np.float32)
        undistorted_pts = cv2.perspectiveTransform(field_pts_arr, self.inv_homography)
        undistorted_pts_2d = undistorted_pts.reshape(-1, 2)

        # 2. Rotation rückgängig machen (POSITIVER Winkel)
        if abs(self.params.rotation) > 0.01:
            center = np.array([w / 2, h / 2])
            angle_rad = np.radians(self.params.rotation)  # POSITIV
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotated_pts = np.zeros_like(undistorted_pts_2d)
            for i, pt in enumerate(undistorted_pts_2d):
                translated = pt - center
                rotated_pts[i, 0] = translated[0] * cos_a - translated[1] * sin_a + center[0]
                rotated_pts[i, 1] = translated[0] * sin_a + translated[1] * cos_a + center[1]
            undistorted_pts_2d = rotated_pts

        # 3. Von gezoomter Kameramatrix zu normalisierten Koordinaten
        fx_zoom = self.zoomed_K[0, 0]
        fy_zoom = self.zoomed_K[1, 1]
        normalized = np.zeros_like(undistorted_pts_2d)
        normalized[:, 0] = (undistorted_pts_2d[:, 0] - cx) / fx_zoom
        normalized[:, 1] = (undistorted_pts_2d[:, 1] - cy) / fy_zoom

        # 4. Fisheye-Verzerrung anwenden
        pts_3d = np.zeros((len(normalized), 3), dtype=np.float32)
        pts_3d[:, 0] = normalized[:, 0]
        pts_3d[:, 1] = normalized[:, 1]
        pts_3d[:, 2] = 1.0

        dist_coeffs_arr = self.params.dist_coeffs.reshape(4, 1)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)

        distorted, _ = cv2.fisheye.projectPoints(
            pts_3d.reshape(-1, 1, 3),
            rvec,
            tvec,
            self.params.camera_matrix,
            dist_coeffs_arr
        )

        return distorted.reshape(-1, 2)

    @staticmethod
    def generate_line_points(
        start: List[float],
        end: List[float],
        segment_length: float = 0.5
    ) -> List[List[float]]:
        """
        Generiert Punkte entlang einer Linie.

        Args:
            start: Startpunkt [x, y]
            end: Endpunkt [x, y]
            segment_length: Abstand zwischen Punkten

        Returns:
            Liste von Punkten
        """
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_segments = max(int(dist / segment_length), 2)
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            points.append([x, y])
        return points

    def generate_grid_lines(
        self,
        segment_length: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generiert alle Spielfeld-Linien als verzerrte Pixel-Koordinaten.

        Args:
            segment_length: Abstand zwischen Punkten für glatte Kurven

        Returns:
            Liste von Grid-Linien mit Typ und Punkten
        """
        L = self.FIELD_LENGTH
        W = self.FIELD_WIDTH
        grid_lines = []

        # Spielfeld-Rand (4 Seiten)
        # Untere Seite
        bottom_pts = self.generate_line_points([0, 0], [L, 0], segment_length)
        bottom_pixels = self.field_to_distorted_pixel(bottom_pts)
        grid_lines.append({"type": "boundary", "points": bottom_pixels.tolist()})

        # Rechte Seite
        right_pts = self.generate_line_points([L, 0], [L, W], segment_length)
        right_pixels = self.field_to_distorted_pixel(right_pts)
        grid_lines.append({"type": "boundary", "points": right_pixels.tolist()})

        # Obere Seite
        top_pts = self.generate_line_points([L, W], [0, W], segment_length)
        top_pixels = self.field_to_distorted_pixel(top_pts)
        grid_lines.append({"type": "boundary", "points": top_pixels.tolist()})

        # Linke Seite
        left_pts = self.generate_line_points([0, W], [0, 0], segment_length)
        left_pixels = self.field_to_distorted_pixel(left_pts)
        grid_lines.append({"type": "boundary", "points": left_pixels.tolist()})

        # Mittellinie
        midline_pts = self.generate_line_points([L/2, 0], [L/2, W], segment_length)
        midline_pixels = self.field_to_distorted_pixel(midline_pts)
        grid_lines.append({"type": "midline", "points": midline_pixels.tolist()})

        # Torlinien (2.85m von der Endlinie)
        goal_dist = self.GOAL_LINE_DIST

        left_goal_pts = self.generate_line_points([goal_dist, 0], [goal_dist, W], segment_length)
        left_goal_pixels = self.field_to_distorted_pixel(left_goal_pts)
        grid_lines.append({"type": "goal_line", "points": left_goal_pixels.tolist()})

        right_goal_pts = self.generate_line_points([L - goal_dist, 0], [L - goal_dist, W], segment_length)
        right_goal_pixels = self.field_to_distorted_pixel(right_goal_pts)
        grid_lines.append({"type": "goal_line", "points": right_goal_pixels.tolist()})

        # Mittelkreis (3m Radius)
        center_x, center_y = L/2, W/2
        circle_points = []
        for angle in range(0, 361, 6):  # 6 Grad Schritte
            rad = np.radians(angle)
            circle_points.append([
                center_x + self.CENTER_CIRCLE_RADIUS * np.cos(rad),
                center_y + self.CENTER_CIRCLE_RADIUS * np.sin(rad)
            ])
        circle_pixels = self.field_to_distorted_pixel(circle_points)
        grid_lines.append({"type": "center_circle", "points": circle_pixels.tolist()})

        # Mittelpunkt
        center_pixel = self.field_to_distorted_pixel([[center_x, center_y]])
        grid_lines.append({"type": "center_point", "points": center_pixel.tolist()})

        # Torraum-Halbkreise (4m Radius)
        goal_radius = self.CREASE_RADIUS

        # Linker Torraum
        left_crease = []
        for angle in range(-90, 91, 6):
            rad = np.radians(angle)
            left_crease.append([
                goal_dist + goal_radius * np.cos(rad),
                center_y + goal_radius * np.sin(rad)
            ])
        left_crease_pixels = self.field_to_distorted_pixel(left_crease)
        grid_lines.append({"type": "crease", "points": left_crease_pixels.tolist()})

        # Rechter Torraum
        right_crease = []
        for angle in range(90, 271, 6):
            rad = np.radians(angle)
            right_crease.append([
                L - goal_dist + goal_radius * np.cos(rad),
                center_y + goal_radius * np.sin(rad)
            ])
        right_crease_pixels = self.field_to_distorted_pixel(right_crease)
        grid_lines.append({"type": "crease", "points": right_crease_pixels.tolist()})

        return grid_lines
