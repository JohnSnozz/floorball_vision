"""
Calibration Processing

Homography-Berechnung und Koordinaten-Transformation für Spielfeld-Kalibrierung.
Unterstützt Fisheye-Entzerrung mit Gyroflow-kompatiblen Lens-Profilen.
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Projekt-Root für Lens-Profile
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LENS_PROFILES_DIR = PROJECT_ROOT / "data" / "lens_profiles"


# ============================================================================
# Lens Profile Management
# ============================================================================

def get_available_lens_profiles() -> List[Dict[str, Any]]:
    """
    Gibt alle verfügbaren Lens-Profile zurück.

    Returns:
        Liste von Profile-Infos mit id, name, description
    """
    profiles = []

    if not LENS_PROFILES_DIR.exists():
        return profiles

    for profile_file in LENS_PROFILES_DIR.glob("*.json"):
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
                profiles.append({
                    "id": profile_file.stem,
                    "name": data.get("name", profile_file.stem),
                    "description": data.get("description", ""),
                    "camera_model": data.get("camera_model", ""),
                    "lens_model": data.get("lens_model", ""),
                    "resolution": data.get("resolution", {})
                })
        except Exception as e:
            print(f"Fehler beim Laden von {profile_file}: {e}")

    return profiles


def load_lens_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Lädt ein Lens-Profil.

    Args:
        profile_id: ID des Profils (Dateiname ohne .json)

    Returns:
        Profil-Daten oder None
    """
    profile_path = LENS_PROFILES_DIR / f"{profile_id}.json"

    if not profile_path.exists():
        return None

    with open(profile_path, 'r') as f:
        return json.load(f)


# ============================================================================
# Fisheye Undistortion
# ============================================================================

def get_fisheye_maps(
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    image_size: Tuple[int, int],
    new_camera_matrix: Optional[np.ndarray] = None,
    balance: float = 0.0,
    zoom_out: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Berechnet die Undistortion-Maps für Fisheye-Korrektur.

    Args:
        camera_matrix: 3x3 Kameramatrix
        distortion_coeffs: 4 Fisheye-Koeffizienten [k1, k2, k3, k4]
        image_size: (width, height) des Bildes
        new_camera_matrix: Optionale neue Kameramatrix für Output
        balance: 0 = alle Pixel gültig, 1 = alle Source-Pixel erhalten
        zoom_out: Faktor > 1.0 vergrößert das Ausgabebild um mehr vom entzerrten Bild zu zeigen

    Returns:
        (map1, map2, new_size) für cv2.remap()
    """
    width, height = image_size

    # Sicherstellen dass distortion_coeffs das richtige Format hat (4,1) für OpenCV fisheye
    dist = np.array(distortion_coeffs, dtype=np.float64)
    if dist.ndim == 1:
        dist = dist.reshape(4, 1)

    # Sicherstellen dass camera_matrix float64 ist
    K = np.array(camera_matrix, dtype=np.float64)

    # Berechne neue Ausgabegröße wenn zoom_out > 1
    new_width = int(width * zoom_out)
    new_height = int(height * zoom_out)
    new_size = (new_width, new_height)

    if new_camera_matrix is None:
        # Neue Kameramatrix berechnen die alle Pixel behält
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            dist,
            (width, height),
            np.eye(3),
            balance=balance,
            new_size=new_size
        )

    # Undistortion maps berechnen
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        dist,
        np.eye(3),
        new_camera_matrix,
        new_size,
        cv2.CV_16SC2
    )

    return map1, map2, new_size


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    balance: float = 0.5,
    zoom_out: float = 1.0
) -> np.ndarray:
    """
    Entzerrt ein Fisheye-Bild.

    Args:
        image: Eingabebild
        camera_matrix: 3x3 Kameramatrix
        distortion_coeffs: 4 Fisheye-Koeffizienten
        balance: 0 = alle Pixel gültig, 1 = alle Source-Pixel erhalten
        zoom_out: Faktor > 1.0 vergrößert das Ausgabebild um mehr vom entzerrten Bild zu zeigen

    Returns:
        Entzerrtes Bild
    """
    h, w = image.shape[:2]

    map1, map2, new_size = get_fisheye_maps(
        camera_matrix,
        distortion_coeffs,
        (w, h),
        balance=balance,
        zoom_out=zoom_out
    )

    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def undistort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray
) -> np.ndarray:
    """
    Entzerrt Punktkoordinaten (Fisheye → Normalized).

    Args:
        points: Nx2 Array von Punkten in Pixel-Koordinaten
        camera_matrix: 3x3 Kameramatrix
        distortion_coeffs: 4 Fisheye-Koeffizienten

    Returns:
        Nx2 Array von entzerrten Punkten
    """
    # Reshape für OpenCV
    pts = points.reshape(-1, 1, 2).astype(np.float32)

    # Sicherstellen dass distortion_coeffs das richtige Format hat (4,1) für OpenCV fisheye
    dist = np.array(distortion_coeffs, dtype=np.float64)
    if dist.ndim == 1:
        dist = dist.reshape(4, 1)

    # Sicherstellen dass camera_matrix float64 ist
    K = np.array(camera_matrix, dtype=np.float64)

    # Undistort mit Fisheye-Modell
    undistorted = cv2.fisheye.undistortPoints(
        pts,
        K,
        dist,
        P=K  # Zurück in Pixel-Koordinaten projizieren
    )

    return undistorted.reshape(-1, 2)


def distort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray
) -> np.ndarray:
    """
    Wendet Fisheye-Verzerrung auf Punkte an (Inverse von undistort).
    Verwendet iterative Approximation.

    Args:
        points: Nx2 Array von entzerrten Punkten
        camera_matrix: 3x3 Kameramatrix
        distortion_coeffs: 4 Fisheye-Koeffizienten

    Returns:
        Nx2 Array von verzerrten Punkten
    """
    # Fisheye hat keine direkte distort-Funktion, wir nutzen projectPoints
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Punkte zu normalisierten Koordinaten
    normalized = np.zeros_like(points)
    normalized[:, 0] = (points[:, 0] - cx) / fx
    normalized[:, 1] = (points[:, 1] - cy) / fy

    # 3D-Punkte auf Z=1 Ebene
    pts_3d = np.zeros((len(points), 3), dtype=np.float32)
    pts_3d[:, 0] = normalized[:, 0]
    pts_3d[:, 1] = normalized[:, 1]
    pts_3d[:, 2] = 1.0

    # Projizieren mit Verzerrung
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)

    distorted, _ = cv2.fisheye.projectPoints(
        pts_3d.reshape(-1, 1, 3),
        rvec,
        tvec,
        camera_matrix,
        distortion_coeffs
    )

    return distorted.reshape(-1, 2)


def scale_camera_matrix(
    camera_matrix: np.ndarray,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Skaliert die Kameramatrix für eine andere Auflösung.

    Args:
        camera_matrix: Original 3x3 Kameramatrix
        original_size: (width, height) der Kalibrierung
        target_size: (width, height) des Zielbildes

    Returns:
        Skalierte Kameramatrix
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled = camera_matrix.copy()
    scaled[0, 0] *= scale_x  # fx
    scaled[1, 1] *= scale_y  # fy
    scaled[0, 2] *= scale_x  # cx
    scaled[1, 2] *= scale_y  # cy

    return scaled


# ============================================================================
# Distortion Estimation from Straight Lines
# ============================================================================

def estimate_distortion_from_lines(
    straight_lines: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    initial_k1: float = 0.0
) -> Dict[str, Any]:
    """
    Schätzt Fisheye-Distortion-Parameter aus Linien die in der Realität gerade sind.

    Die Idee: Wenn eine Linie in der Realität gerade ist, aber im Bild gekrümmt erscheint,
    dann ist die Krümmung ein Mass für die Verzerrung. Wir optimieren k1 (den dominanten
    Distortion-Koeffizienten) so, dass die entzerrten Linien möglichst gerade werden.

    Args:
        straight_lines: Liste von Linien-Dicts mit:
            - points: Liste von (x, y) Punkten
            - type: "horizontal", "vertical", oder "free"
        image_size: (width, height) des Bildes
        initial_k1: Startwert für k1

    Returns:
        Dict mit geschätzten Parametern:
        - camera_matrix: Geschätzte Kameramatrix
        - distortion_coeffs: [k1, k2, k3, k4]
        - straightness_error: Durchschnittlicher Fehler (niedriger = besser)
    """
    from scipy.optimize import minimize_scalar

    width, height = image_size
    cx, cy = width / 2, height / 2

    # Initiale Brennweite schätzen (typisch für GoPro-artige Kameras)
    focal_length = max(width, height) * 0.8

    def line_straightness_error(points: np.ndarray, line_type: str = "free") -> float:
        """
        Berechnet wie weit Punkte von einer geraden Linie abweichen.

        Args:
            points: Nx2 Array von Punkten
            line_type: "horizontal" (konstantes Y), "vertical" (konstantes X), "free" (beliebig)
        """
        if len(points) < 3:
            return 0.0

        if line_type == "horizontal":
            # Alle Punkte sollten gleiches Y haben
            y_values = points[:, 1]
            mean_y = np.mean(y_values)
            return np.mean(np.abs(y_values - mean_y))

        elif line_type == "vertical":
            # Alle Punkte sollten gleiches X haben
            x_values = points[:, 0]
            mean_x = np.mean(x_values)
            return np.mean(np.abs(x_values - mean_x))

        else:
            # Freie Linie: Distanz zur besten Fit-Linie (SVD)
            centroid = np.mean(points, axis=0)
            centered = points - centroid

            # SVD für Hauptrichtung
            _, _, vh = np.linalg.svd(centered)
            direction = vh[0]  # Hauptrichtung

            # Distanz jedes Punktes zur Linie
            projections = np.outer(centered @ direction, direction) + centroid
            distances = np.linalg.norm(points - projections, axis=1)

            return np.mean(distances)

    def undistort_points_simple(points: np.ndarray, k1: float, fx: float, fy: float) -> np.ndarray:
        """
        Vereinfachte Punkt-Entzerrung mit nur k1 Parameter.
        Verwendet radiale Distortion: r_undist = r_dist * (1 + k1 * r_dist^2)
        """
        # Normalisierte Koordinaten
        x_norm = (points[:, 0] - cx) / fx
        y_norm = (points[:, 1] - cy) / fy

        r_sq = x_norm**2 + y_norm**2

        # Radiale Entzerrung
        factor = 1 + k1 * r_sq

        x_undist = x_norm * factor
        y_undist = y_norm * factor

        # Zurück zu Pixel-Koordinaten
        return np.column_stack([
            x_undist * fx + cx,
            y_undist * fy + cy
        ])

    def total_error(k1: float) -> float:
        """Gesamtfehler über alle Linien"""
        total = 0.0
        count = 0

        for line_data in straight_lines:
            points = line_data.get("points", [])
            line_type = line_data.get("type", "free")

            if len(points) < 3:
                continue

            pts = np.array(points, dtype=np.float64)
            undistorted = undistort_points_simple(pts, k1, focal_length, focal_length)
            error = line_straightness_error(undistorted, line_type)

            # Gewichtung: Horizontale/vertikale Linien sind wichtiger
            weight = 1.5 if line_type in ["horizontal", "vertical"] else 1.0
            total += error * weight
            count += weight

        return total / max(count, 1)

    # Optimiere k1 im Bereich [-0.5, 0.5]
    result = minimize_scalar(
        total_error,
        bounds=(-0.5, 0.5),
        method='bounded',
        options={'xatol': 0.001}
    )

    best_k1 = result.x
    best_error = result.fun

    # Kameramatrix erstellen
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Distortion-Koeffizienten (nur k1 geschätzt, Rest auf 0)
    distortion_coeffs = np.array([best_k1, 0.0, 0.0, 0.0], dtype=np.float64)

    return {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coeffs": distortion_coeffs.tolist(),
        "estimated_k1": best_k1,
        "straightness_error": best_error,
        "focal_length": focal_length,
        "principal_point": [cx, cy],
        "image_size": {"width": width, "height": height}
    }


def save_estimated_profile(
    estimation_result: Dict[str, Any],
    profile_name: str,
    description: str = ""
) -> str:
    """
    Speichert geschätzte Distortion-Parameter als neues Lens-Profil.

    Args:
        estimation_result: Ergebnis von estimate_distortion_from_lines()
        profile_name: Name für das Profil (wird auch als Dateiname verwendet)
        description: Optionale Beschreibung

    Returns:
        Pfad zum gespeicherten Profil
    """
    # Profil-ID aus Name ableiten
    profile_id = profile_name.lower().replace(" ", "_").replace("-", "_")
    profile_path = LENS_PROFILES_DIR / f"{profile_id}.json"

    # Profil-Daten
    profile_data = {
        "name": profile_name,
        "description": description or f"Geschätzt aus Hilfslinien. k1={estimation_result['estimated_k1']:.4f}",
        "camera_brand": "Geschätzt",
        "camera_model": "Aus Hilfslinien",
        "lens_model": "Unbekannt",
        "resolution": estimation_result["image_size"],
        "fisheye_params": {
            "camera_matrix": estimation_result["camera_matrix"],
            "distortion_coeffs": estimation_result["distortion_coeffs"]
        },
        "source": "Geschätzt aus manuell gezeichneten Hilfslinien",
        "estimated": True,
        "straightness_error": estimation_result["straightness_error"]
    }

    # Verzeichnis erstellen falls nötig
    LENS_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # Speichern
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=2)

    return str(profile_path)


def extract_calibration_frame(
    video_path: str,
    timestamp: float = 10.0,
    output_dir: str = "data/calibration"
) -> str:
    """
    Extrahiert einen Frame aus dem Video für die Kalibrierung.

    Args:
        video_path: Pfad zum Video
        timestamp: Zeit in Sekunden
        output_dir: Ausgabe-Verzeichnis

    Returns:
        Pfad zum extrahierten Frame
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "calibration_frame.jpg")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video konnte nicht geöffnet werden: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Frame bei {timestamp}s konnte nicht gelesen werden")

    cv2.imwrite(output_path, frame)
    return output_path


def extract_multiple_screenshots(
    video_path: str,
    output_dir: str,
    num_screenshots: int = 20,
    skip_start_percent: float = 5.0,
    skip_end_percent: float = 5.0
) -> List[dict]:
    """
    Extrahiert mehrere Screenshots gleichmässig verteilt über das Video.

    Args:
        video_path: Pfad zum Video
        output_dir: Ausgabe-Verzeichnis
        num_screenshots: Anzahl Screenshots (default: 20)
        skip_start_percent: Prozent am Anfang überspringen (default: 5%)
        skip_end_percent: Prozent am Ende überspringen (default: 5%)

    Returns:
        Liste von dicts mit {filename, timestamp, path}
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video konnte nicht geöffnet werden: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Start/Ende überspringen
    start_time = duration * (skip_start_percent / 100)
    end_time = duration * (1 - skip_end_percent / 100)
    usable_duration = end_time - start_time

    if usable_duration <= 0:
        cap.release()
        raise ValueError("Video zu kurz für Screenshot-Extraktion")

    # Zeitpunkte berechnen
    interval = usable_duration / (num_screenshots - 1) if num_screenshots > 1 else 0
    timestamps = [start_time + i * interval for i in range(num_screenshots)]

    screenshots = []

    for i, timestamp in enumerate(timestamps):
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            filename = f"screenshot_{i+1:02d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)

            screenshots.append({
                "index": i + 1,
                "filename": filename,
                "timestamp": round(timestamp, 2),
                "timestamp_formatted": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                "path": output_path
            })

    cap.release()

    return screenshots


def get_existing_screenshots(output_dir: str) -> List[dict]:
    """
    Lädt Informationen über bereits existierende Screenshots.

    Args:
        output_dir: Verzeichnis mit Screenshots

    Returns:
        Liste von dicts mit {index, filename, timestamp, timestamp_formatted, path}
    """
    screenshots = []

    if not os.path.exists(output_dir):
        return screenshots

    # Alle screenshot_XX.jpg Dateien finden
    import re
    pattern = re.compile(r'screenshot_(\d+)\.jpg')

    for filename in sorted(os.listdir(output_dir)):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            filepath = os.path.join(output_dir, filename)

            # Timestamp ist nicht mehr bekannt, verwende Platzhalter
            screenshots.append({
                "index": index,
                "filename": filename,
                "timestamp": 0,  # Unbekannt
                "timestamp_formatted": f"#{index}",
                "path": filepath
            })

    return screenshots


def calculate_homography(
    image_points: List[List[float]],
    field_points: List[List[float]]
) -> np.ndarray:
    """
    Berechnet die Homography-Matrix für die Transformation.

    Args:
        image_points: Liste von [x, y] Pixel-Koordinaten im Bild
        field_points: Liste von [x, y] Koordinaten auf dem Spielfeld (Meter)

    Returns:
        3x3 Homography-Matrix
    """
    if len(image_points) != len(field_points):
        raise ValueError("Anzahl der Punkte muss übereinstimmen")

    if len(image_points) < 4:
        raise ValueError("Mindestens 4 Punkt-Paare erforderlich")

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(field_points, dtype=np.float32)

    # Homography berechnen mit RANSAC für Robustheit
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if matrix is None:
        raise ValueError("Homography-Matrix konnte nicht berechnet werden")

    return matrix


def calculate_combined_calibration(
    image_points: List[List[float]],
    field_points: List[List[float]],
    image_size: Tuple[int, int],
    optimize_distortion: bool = True
) -> Dict[str, Any]:
    """
    Berechnet gleichzeitig Fisheye-Korrektur UND Homographie aus Referenzpunkten.

    Diese Funktion optimiert beide Parameter zusammen, so dass die Transformation
    von verzerrten Bild-Koordinaten zu Feld-Koordinaten möglichst genau ist.

    Args:
        image_points: Liste von [x, y] Pixel-Koordinaten im verzerrten Bild
        field_points: Liste von [x, y] Koordinaten auf dem Spielfeld (Meter)
        image_size: (width, height) des Bildes
        optimize_distortion: Wenn True, wird k1 optimiert, sonst k1=0

    Returns:
        Dict mit:
            - homography_matrix: 3x3 Matrix
            - k1: Radiale Verzerrung (negativ = barrel/fisheye)
            - camera_matrix: 3x3 Kameramatrix
            - distortion_coeffs: [k1, 0, 0, 0]
            - reprojection_error: Durchschnittlicher Fehler in Metern
    """
    from scipy.optimize import minimize_scalar, minimize

    if len(image_points) != len(field_points):
        raise ValueError("Anzahl der Punkte muss übereinstimmen")

    if len(image_points) < 4:
        raise ValueError("Mindestens 4 Punkt-Paare erforderlich")

    width, height = image_size
    cx, cy = width / 2, height / 2

    # Focal length schätzen (typisch für Weitwinkel-Kameras)
    focal = max(width, height) * 0.8

    src_points = np.array(image_points, dtype=np.float64)
    dst_points = np.array(field_points, dtype=np.float64)

    def undistort_points_simple(points, k1, cx, cy, focal):
        """Entzerrt Punkte mit einfachem radialen Modell."""
        undistorted = []
        for px, py in points:
            # Normalisierte Koordinaten (relativ zum Zentrum)
            x = (px - cx) / focal
            y = (py - cy) / focal
            r2 = x*x + y*y

            # Radiale Entzerrung: r_undist = r_dist * (1 + k1*r^2)
            # Für Fisheye ist k1 typischerweise negativ
            factor = 1 + k1 * r2

            # Zurück zu Pixel-Koordinaten
            x_undist = x * factor * focal + cx
            y_undist = y * factor * focal + cy
            undistorted.append([x_undist, y_undist])

        return np.array(undistorted, dtype=np.float64)

    def compute_error(k1):
        """Berechnet den Gesamtfehler für einen gegebenen k1-Wert."""
        # Punkte entzerren
        undist_points = undistort_points_simple(src_points, k1, cx, cy, focal)

        # Homographie aus entzerrten Punkten berechnen
        try:
            H, _ = cv2.findHomography(undist_points.astype(np.float32),
                                      dst_points.astype(np.float32),
                                      cv2.RANSAC, 5.0)
            if H is None:
                return float('inf')

            # Reprojektions-Fehler berechnen
            pts = undist_points.reshape(-1, 1, 2).astype(np.float32)
            projected = cv2.perspectiveTransform(pts, H)
            projected = projected.reshape(-1, 2)

            errors = np.sqrt(np.sum((projected - dst_points)**2, axis=1))
            return np.mean(errors)
        except:
            return float('inf')

    # Optimierung
    if optimize_distortion:
        # Erweiterte Optimierung: k1 UND focal gleichzeitig optimieren
        def compute_error_2d(params):
            k1_val, focal_scale = params
            actual_focal = focal * focal_scale
            undist = undistort_points_simple(src_points, k1_val, cx, cy, actual_focal)
            try:
                H, _ = cv2.findHomography(undist.astype(np.float32),
                                          dst_points.astype(np.float32),
                                          cv2.RANSAC, 5.0)
                if H is None:
                    return float('inf')
                pts = undist.reshape(-1, 1, 2).astype(np.float32)
                projected = cv2.perspectiveTransform(pts, H)
                projected = projected.reshape(-1, 2)
                errors = np.sqrt(np.sum((projected - dst_points)**2, axis=1))
                return np.mean(errors)
            except:
                return float('inf')

        # Grobe Grid-Suche über k1 und focal_scale (reduziert für Geschwindigkeit)
        best_error = float('inf')
        best_k1 = 0.0
        best_focal_scale = 1.0

        # k1: -0.6 bis +0.4 (erweiterter Bereich)
        # focal_scale: 0.4 bis 1.5 (Brennweite kann stark variieren)
        for k1_try in np.linspace(-0.6, 0.4, 15):  # 15 statt 30
            for focal_try in np.linspace(0.4, 1.5, 10):  # 10 statt 15
                err = compute_error_2d([k1_try, focal_try])
                if err < best_error:
                    best_error = err
                    best_k1 = k1_try
                    best_focal_scale = focal_try

        # Feinoptimierung mit scipy (präziser)
        from scipy.optimize import minimize as scipy_minimize
        result = scipy_minimize(
            compute_error_2d,
            [best_k1, best_focal_scale],
            method='Nelder-Mead',
            options={'xatol': 0.0001, 'fatol': 0.0001, 'maxiter': 200}
        )
        best_k1, best_focal_scale = result.x
        focal = focal * best_focal_scale
    else:
        best_k1 = 0.0

    # Finale Berechnung mit optimalem k1
    undist_points = undistort_points_simple(src_points, best_k1, cx, cy, focal)

    H, mask = cv2.findHomography(undist_points.astype(np.float32),
                                  dst_points.astype(np.float32),
                                  cv2.RANSAC, 5.0)

    # Fehler berechnen
    pts = undist_points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(pts, H)
    projected = projected.reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - dst_points)**2, axis=1))

    # Kameramatrix erstellen
    camera_matrix = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return {
        "homography_matrix": H.tolist(),
        "k1": float(best_k1),
        "focal_length": float(focal),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coeffs": [float(best_k1), 0.0, 0.0, 0.0],
        "image_center": [float(cx), float(cy)],
        "reprojection_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "point_errors": errors.tolist(),
        "inlier_mask": mask.flatten().tolist() if mask is not None else None
    }


def transform_point_with_distortion(
    point: Tuple[float, float],
    k1: float,
    focal: float,
    cx: float,
    cy: float,
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Transformiert einen Punkt vom verzerrten Bild zu Feld-Koordinaten.

    1. Entzerrt den Punkt (Fisheye-Korrektur)
    2. Wendet die Homographie an

    Args:
        point: (x, y) Pixel-Koordinate im verzerrten Bild
        k1: Radiale Verzerrung
        focal: Brennweite
        cx, cy: Bildmittelpunkt
        homography_matrix: 3x3 Homographie-Matrix

    Returns:
        (x, y) Feld-Koordinate in Metern
    """
    px, py = point

    # Entzerrung
    x = (px - cx) / focal
    y = (py - cy) / focal
    r2 = x*x + y*y
    factor = 1 + k1 * r2
    x_undist = x * factor * focal + cx
    y_undist = y * factor * focal + cy

    # Homographie
    pts = np.array([[[x_undist, y_undist]]], dtype=np.float32)
    H = np.array(homography_matrix, dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, H)

    return tuple(transformed[0, 0])


def transform_points(
    points: List[List[float]],
    homography_matrix: List[List[float]]
) -> List[List[float]]:
    """
    Transformiert Pixel-Koordinaten zu Spielfeld-Koordinaten.

    Args:
        points: Liste von [x, y] Pixel-Koordinaten
        homography_matrix: 3x3 Homography-Matrix

    Returns:
        Liste von [x, y] Spielfeld-Koordinaten (Meter)
    """
    if not points:
        return []

    # Zu numpy arrays konvertieren
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    matrix = np.array(homography_matrix, dtype=np.float32)

    # Transformation durchführen
    transformed = cv2.perspectiveTransform(pts, matrix)

    # Zurück zu Liste konvertieren
    return transformed.reshape(-1, 2).tolist()


def transform_single_point(
    point: Tuple[float, float],
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Transformiert einen einzelnen Punkt.

    Args:
        point: (x, y) Pixel-Koordinate
        homography_matrix: 3x3 Homography-Matrix

    Returns:
        (x, y) Spielfeld-Koordinate (Meter)
    """
    pts = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, homography_matrix)
    return tuple(transformed[0, 0])


def points_inside_polygon(
    points: List[List[float]],
    polygon: List[List[float]]
) -> List[bool]:
    """
    Prüft ob Punkte innerhalb eines Polygons liegen.

    Args:
        points: Liste von [x, y] Koordinaten
        polygon: Liste von [x, y] Polygon-Eckpunkten

    Returns:
        Liste von Booleans
    """
    if not polygon or len(polygon) < 3:
        # Kein Polygon definiert - alle Punkte sind "inside"
        return [True] * len(points)

    poly = np.array(polygon, dtype=np.float32)
    results = []

    for point in points:
        pt = tuple(map(float, point))
        # pointPolygonTest: > 0 = inside, = 0 = on edge, < 0 = outside
        result = cv2.pointPolygonTest(poly, pt, False)
        results.append(result >= 0)

    return results


def point_inside_polygon(
    point: Tuple[float, float],
    polygon: List[List[float]]
) -> bool:
    """
    Prüft ob ein einzelner Punkt innerhalb des Polygons liegt.

    Args:
        point: (x, y) Koordinate
        polygon: Liste von [x, y] Polygon-Eckpunkten

    Returns:
        True wenn innerhalb oder auf Rand
    """
    if not polygon or len(polygon) < 3:
        return True

    poly = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, tuple(map(float, point)), False)
    return result >= 0


def generate_test_overlay(
    screenshot_path: str,
    calibration_data: dict,
    output_path: str,
    grid_spacing_meters: float = 5.0,
    lens_profile: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generiert ein Test-Overlay mit Grid, Toren und Banden-Polygon.
    Bei Fisheye-Profil werden gekrümmte Linien gezeichnet.

    Floorball-Spielfeld nach IFF-Regeln:
    - Feldgrösse: 40m x 20m
    - Tor: 1.6m breit, 1.15m hoch, 0.65m tief
    - Tor-Abstand zur Bande: 2.85m
    - Torraum: Halbkreis mit 4m Radius
    - Mittelkreis: 3m Radius

    Args:
        screenshot_path: Pfad zum Screenshot
        calibration_data: Kalibrierungsdaten mit homography_matrix und boundary_polygon
        output_path: Ausgabe-Pfad
        grid_spacing_meters: Abstand der Grid-Linien in Metern
        lens_profile: Optionales Lens-Profil für Fisheye-Korrektur

    Returns:
        Pfad zum Overlay-Bild
    """
    # Bild laden
    img = cv2.imread(screenshot_path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {screenshot_path}")

    h, w = img.shape[:2]

    # Prüfe ob kombinierte Kalibrierung vorhanden (mit eigenem k1)
    combined_cal = calibration_data.get("combined_calibration")
    use_combined = combined_cal is not None

    # Homography-Matrix laden
    if use_combined:
        h_matrix = np.array(combined_cal["homography_matrix"], dtype=np.float32)
    else:
        h_matrix = np.array(calibration_data["homography_matrix"], dtype=np.float32)

    h_inv = np.linalg.inv(h_matrix)

    # Fisheye-Parameter laden
    # Priorität: 1. combined_calibration, 2. lens_profile
    camera_matrix = None
    distortion_coeffs = None
    k1 = 0.0
    focal = None
    cx, cy = w / 2, h / 2

    if use_combined and combined_cal.get("k1", 0) != 0:
        # Kombinierte Kalibrierung mit eigenem k1 verwenden
        k1 = combined_cal["k1"]
        focal = combined_cal.get("focal_length", max(w, h) * 0.8)
        cx, cy = combined_cal.get("image_center", [w / 2, h / 2])
        # Wir verwenden das einfache radiale Modell direkt
        camera_matrix = None  # Signal: einfaches Modell verwenden
        distortion_coeffs = None

    elif lens_profile and "fisheye_params" in lens_profile:
        fp = lens_profile["fisheye_params"]
        camera_matrix = np.array(fp["camera_matrix"], dtype=np.float64)
        distortion_coeffs = np.array(fp["distortion_coeffs"], dtype=np.float64)

        # Kameramatrix skalieren falls nötig
        profile_res = lens_profile.get("resolution", {})
        if profile_res.get("width") and profile_res.get("height"):
            if profile_res["width"] != w or profile_res["height"] != h:
                camera_matrix = scale_camera_matrix(
                    camera_matrix,
                    (profile_res["width"], profile_res["height"]),
                    (w, h)
                )

    # Floorball Spielfeld-Dimensionen (IFF Regeln)
    field_length = 40.0
    field_width = 20.0
    goal_distance = 2.85      # Abstand Tor zur Bande
    goal_width = 1.6          # Torbreite
    goal_area_radius = 4.0    # Torraum-Radius (Halbkreis)
    center_circle_radius = 3.0  # Mittelkreis-Radius
    mid_y = field_width / 2

    # Hilfsfunktion: Feld-Koordinate zu Bild transformieren
    # Bei Fisheye: erst Homography, dann Verzerrung anwenden
    def field_to_image(x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        undistorted = cv2.perspectiveTransform(pt, h_inv)[0, 0]

        # Fall 1: Kombinierte Kalibrierung mit einfachem k1-Modell
        if k1 != 0 and focal is not None:
            # Umkehrung der Entzerrung = Verzerrung anwenden
            # Entzerrung war: x_undist = x_dist * (1 + k1 * r_dist^2)
            # Wir müssen x_dist aus x_undist finden (iterativ)
            ux = (undistorted[0] - cx) / focal
            uy = (undistorted[1] - cy) / focal

            # Newton-Raphson Iteration um x_dist zu finden
            # Start mit Approximation
            dx, dy = ux, uy
            for _ in range(10):  # Max 10 Iterationen
                r2 = dx*dx + dy*dy
                factor = 1 + k1 * r2
                # Die Gleichung ist: undist = dist * factor
                # Also: dist_new = undist / factor (basierend auf aktuellem r)
                if abs(factor) > 0.01:
                    dx_new = ux / factor
                    dy_new = uy / factor
                    # Konvergenz-Check
                    if abs(dx_new - dx) < 1e-6 and abs(dy_new - dy) < 1e-6:
                        dx, dy = dx_new, dy_new
                        break
                    dx, dy = dx_new, dy_new
                else:
                    break

            distorted_x = dx * focal + cx
            distorted_y = dy * focal + cy
            return np.array([distorted_x, distorted_y])

        # Fall 2: Lens-Profil mit OpenCV Fisheye-Modell
        elif camera_matrix is not None and distortion_coeffs is not None:
            # Fisheye-Verzerrung anwenden
            distorted = distort_points(
                undistorted.reshape(1, 2),
                camera_matrix,
                distortion_coeffs
            )
            return distorted[0]

        return undistorted

    # Hilfsfunktion: Mehrere Punkte entlang einer Linie erzeugen (für gekrümmte Linien)
    def field_line_to_image_points(x1, y1, x2, y2, num_segments=20):
        """Erzeugt interpolierte Punkte für eine Linie auf dem Spielfeld."""
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            points.append(field_to_image(x, y))
        return np.array(points, dtype=np.int32)

    # Grid zeichnen
    overlay = img.copy()

    # Anzahl Segmente für gekrümmte Linien (mehr bei Fisheye oder k1)
    use_curved_lines = (camera_matrix is not None) or (k1 != 0)
    num_segments = 30 if use_curved_lines else 1

    # Vertikale Linien (alle X Meter)
    for x in np.arange(0, field_length + 0.1, grid_spacing_meters):
        # Mittellinie dicker
        thickness = 3 if abs(x - field_length / 2) < 0.1 else 1
        color = (0, 255, 255) if abs(x - field_length / 2) < 0.1 else (0, 255, 0)

        if num_segments > 1:
            # Gekrümmte Linie zeichnen
            pts = field_line_to_image_points(x, 0, x, field_width, num_segments)
            cv2.polylines(overlay, [pts], False, color, thickness)
        else:
            # Gerade Linie
            pt1 = tuple(map(int, field_to_image(x, 0)))
            pt2 = tuple(map(int, field_to_image(x, field_width)))
            cv2.line(overlay, pt1, pt2, color, thickness)

    # Horizontale Linien (alle X Meter)
    for y in np.arange(0, field_width + 0.1, grid_spacing_meters):
        # Mittellinie dicker
        thickness = 3 if abs(y - field_width / 2) < 0.1 else 1
        color = (0, 255, 255) if abs(y - field_width / 2) < 0.1 else (0, 255, 0)

        if num_segments > 1:
            # Gekrümmte Linie zeichnen
            pts = field_line_to_image_points(0, y, field_length, y, num_segments)
            cv2.polylines(overlay, [pts], False, color, thickness)
        else:
            # Gerade Linie
            pt1 = tuple(map(int, field_to_image(0, y)))
            pt2 = tuple(map(int, field_to_image(field_length, y)))
            cv2.line(overlay, pt1, pt2, color, thickness)

    # Spielfeld-Rand (rot) - mit gekrümmten Linien bei Fisheye
    if num_segments > 1:
        # Untere Kante (y=0)
        pts = field_line_to_image_points(0, 0, field_length, 0, num_segments)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)
        # Rechte Kante (x=field_length)
        pts = field_line_to_image_points(field_length, 0, field_length, field_width, num_segments)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)
        # Obere Kante (y=field_width)
        pts = field_line_to_image_points(field_length, field_width, 0, field_width, num_segments)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)
        # Linke Kante (x=0)
        pts = field_line_to_image_points(0, field_width, 0, 0, num_segments)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)
    else:
        corners_field = np.array([
            [[0, 0]],
            [[field_length, 0]],
            [[field_length, field_width]],
            [[0, field_width]]
        ], dtype=np.float32)
        corners_img = cv2.perspectiveTransform(corners_field, h_inv)

        for i in range(4):
            pt1 = tuple(map(int, corners_img[i, 0]))
            pt2 = tuple(map(int, corners_img[(i + 1) % 4, 0]))
            cv2.line(overlay, pt1, pt2, (0, 0, 255), 3)

    # Mittelkreis (cyan)
    center_points = []
    for angle in np.linspace(0, 2 * np.pi, 36):
        x = field_length / 2 + center_circle_radius * np.cos(angle)
        y = mid_y + center_circle_radius * np.sin(angle)
        center_points.append(field_to_image(x, y))

    center_pts = np.array(center_points, dtype=np.int32)
    cv2.polylines(overlay, [center_pts], True, (255, 255, 0), 2)

    # Mittelpunkt
    center_img = field_to_image(field_length / 2, mid_y)
    cv2.circle(overlay, tuple(map(int, center_img)), 5, (255, 255, 0), -1)

    # Linkes Tor (magenta)
    tor_left_top = field_to_image(goal_distance, mid_y + goal_width / 2)
    tor_left_bottom = field_to_image(goal_distance, mid_y - goal_width / 2)
    cv2.line(overlay, tuple(map(int, tor_left_top)), tuple(map(int, tor_left_bottom)), (255, 0, 255), 4)

    # Linker Torraum (Halbkreis)
    goal_area_left = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 20):
        x = goal_distance + goal_area_radius * np.cos(angle)
        y = mid_y + goal_area_radius * np.sin(angle)
        goal_area_left.append(field_to_image(x, y))

    goal_area_left_pts = np.array(goal_area_left, dtype=np.int32)
    cv2.polylines(overlay, [goal_area_left_pts], False, (255, 0, 255), 2)

    # Rechtes Tor (magenta)
    tor_right_top = field_to_image(field_length - goal_distance, mid_y + goal_width / 2)
    tor_right_bottom = field_to_image(field_length - goal_distance, mid_y - goal_width / 2)
    cv2.line(overlay, tuple(map(int, tor_right_top)), tuple(map(int, tor_right_bottom)), (255, 0, 255), 4)

    # Rechter Torraum (Halbkreis)
    goal_area_right = []
    for angle in np.linspace(np.pi / 2, 3 * np.pi / 2, 20):
        x = field_length - goal_distance + goal_area_radius * np.cos(angle)
        y = mid_y + goal_area_radius * np.sin(angle)
        goal_area_right.append(field_to_image(x, y))

    goal_area_right_pts = np.array(goal_area_right, dtype=np.int32)
    cv2.polylines(overlay, [goal_area_right_pts], False, (255, 0, 255), 2)

    # === ZUSÄTZLICHE GERADEN für Entzerrungs-Kontrolle ===
    # Diese Linien sollten mit den echten Banden übereinstimmen
    # Wenn sie gekrümmt sind und auf die gekrümmten Banden passen = Entzerrung korrekt

    # Zusätzliche horizontale Linien bei 2m und 18m (nahe an Banden)
    for y_offset in [2.0, 18.0]:
        pts = field_line_to_image_points(0, y_offset, field_length, y_offset, num_segments)
        cv2.polylines(overlay, [pts], False, (255, 128, 0), 1)  # Orange

    # Zusätzliche vertikale Linien bei 2m und 38m (nahe an kurzen Banden)
    for x_offset in [2.0, 38.0]:
        pts = field_line_to_image_points(x_offset, 0, x_offset, field_width, num_segments)
        cv2.polylines(overlay, [pts], False, (255, 128, 0), 1)  # Orange

    # Diagonalen über das gesamte Feld (sollten als Kurven erscheinen)
    pts_diag1 = field_line_to_image_points(0, 0, field_length, field_width, num_segments)
    cv2.polylines(overlay, [pts_diag1], False, (128, 255, 128), 1)  # Hellgrün

    pts_diag2 = field_line_to_image_points(0, field_width, field_length, 0, num_segments)
    cv2.polylines(overlay, [pts_diag2], False, (128, 255, 128), 1)  # Hellgrün

    # Banden-Polygon zeichnen (blau)
    boundary = calibration_data.get("boundary_polygon", [])
    if boundary and len(boundary) >= 3:
        poly_pts = np.array(boundary, dtype=np.int32)
        cv2.polylines(overlay, [poly_pts], True, (255, 0, 0), 2)

    # Referenzpunkte markieren
    image_points = calibration_data.get("image_points", [])
    for i, pt in enumerate(image_points):
        center = tuple(map(int, pt))
        cv2.circle(overlay, center, 8, (0, 165, 255), -1)  # Orange
        cv2.putText(overlay, str(i + 1), (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Mit Original überblenden
    result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Legende hinzufügen
    legend_y = 30
    cv2.putText(result, "Gruen: Grid (5m)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, "Gelb: Mittellinie", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(result, "Rot: Spielfeldrand", (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(result, "Cyan: Mittelkreis", (10, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(result, "Magenta: Tore/Torraum", (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(result, "Blau: Banden-Polygon", (10, legend_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(result, "Orange: Referenzpunkte", (10, legend_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Speichern
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)

    return output_path


def get_field_coordinates_for_bbox(
    bbox: Tuple[float, float, float, float],
    homography_matrix: np.ndarray,
    use_bottom_center: bool = True
) -> Tuple[float, float]:
    """
    Berechnet die Spielfeld-Koordinaten für eine Bounding Box.

    Args:
        bbox: (x1, y1, x2, y2) Bounding Box
        homography_matrix: 3x3 Homography-Matrix
        use_bottom_center: True = untere Mitte (Füsse), False = Zentrum

    Returns:
        (x, y) Spielfeld-Koordinaten in Metern
    """
    x1, y1, x2, y2 = bbox

    if use_bottom_center:
        # Untere Mitte der Box (wo die Füsse sind)
        point = ((x1 + x2) / 2, y2)
    else:
        # Zentrum der Box
        point = ((x1 + x2) / 2, (y1 + y2) / 2)

    return transform_single_point(point, homography_matrix)
