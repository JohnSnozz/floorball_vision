#!/usr/bin/env python3
"""
Generiert Spielfeld-Overlay auf dem entzerrten Bild.

Passe die POINT_PAIRS unten an - Bild-Koordinaten kannst du z.B. mit
einem Bildbearbeitungsprogramm ablesen (Mausposition).

Ausführen: python scripts/generate_overlay.py
"""
import cv2
import numpy as np
from pathlib import Path

# ============================================================================
# KONFIGURATION - HIER ANPASSEN!
# ============================================================================

VIDEO_ID = "340d8ea8-00dc-4415-95da-ae18d54daf6b"

# Punkt-Paare: (Bild-X, Bild-Y) -> (Feld-X, Feld-Y)
# Feld-Koordinaten in Metern:
#   - Spielfeld ist 40m x 20m
#   - (0, 0) = Ecke unten-links (vom Kamera-Blickwinkel aus)
#   - (40, 0) = Ecke unten-rechts
#   - (0, 20) = Ecke oben-links
#   - (40, 20) = Ecke oben-rechts
#   - (20, 10) = Mittelpunkt
#   - (0, 10) = Tor links Mitte
#   - (40, 10) = Tor rechts Mitte

# Punkte basierend auf dem entzerrten Bild (2304 x 1296)
# KORRIGIERT - Banden folgen der tatsächlichen Spielfeld-Begrenzung
POINT_PAIRS = [
    # (Bild-X, Bild-Y, Feld-X, Feld-Y)

    # Mittellinie unten (wo sie die vordere Bande trifft) - weiter unten
    (1152, 820, 20, 0),

    # Mittellinie oben (wo sie die hintere Bande trifft)
    (1152, 295, 20, 20),

    # Linke Ecke vorne - weiter nach links/unten zur Bande
    (95, 720, 0, 0),

    # Rechte Ecke vorne - weiter nach rechts zur Bande
    (2210, 720, 40, 0),

    # Linke Ecke hinten - zur oberen Bande
    (420, 295, 0, 20),

    # Rechte Ecke hinten
    (1885, 295, 40, 20),

    # Mittelpunkt (Mitte des Mittelkreises) - etwas höher
    (1152, 480, 20, 10),

    # Tor links (Mitte des Tors an der Bande)
    (180, 480, 0, 10),

    # Tor rechts (Mitte des Tors an der Bande)
    (2125, 480, 40, 10),
]

# ============================================================================
# AB HIER NICHT ÄNDERN
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration" / VIDEO_ID
IMAGE_PATH = CALIBRATION_DIR / "undistorted_frame.jpg"
OUTPUT_PATH = CALIBRATION_DIR / "manual_overlay.jpg"

FIELD_WIDTH = 40.0
FIELD_HEIGHT = 20.0


def draw_field_overlay(img, homography_inv):
    """Zeichnet das Spielfeld-Overlay auf das Bild."""
    overlay = img.copy()

    # Grid (5m Abstand) - Grün
    for x in range(0, int(FIELD_WIDTH) + 1, 5):
        pts = []
        for y in np.linspace(0, FIELD_HEIGHT, 50):
            field_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
            pts.append(img_pt[0][0])
        pts = np.array(pts, dtype=np.int32)
        color = (0, 255, 255) if x == 20 else (0, 255, 0)  # Gelb für Mittellinie
        thickness = 3 if x == 20 else 2
        cv2.polylines(overlay, [pts], False, color, thickness)

    for y in range(0, int(FIELD_HEIGHT) + 1, 5):
        pts = []
        for x_val in np.linspace(0, FIELD_WIDTH, 100):
            field_pt = np.array([[[x_val, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
            pts.append(img_pt[0][0])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)

    # Spielfeldrand - Rot, dick
    border = [(0, 0), (FIELD_WIDTH, 0), (FIELD_WIDTH, FIELD_HEIGHT), (0, FIELD_HEIGHT), (0, 0)]
    border_pts = []
    for pt in border:
        field_pt = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
        border_pts.append(img_pt[0][0])
    border_pts = np.array(border_pts, dtype=np.int32)
    cv2.polylines(overlay, [border_pts], False, (0, 0, 255), 4)

    # Mittelkreis (3m Radius) - Cyan
    circle_pts = []
    for angle in np.linspace(0, 2 * np.pi, 60):
        cx = 20 + 3 * np.cos(angle)
        cy = 10 + 3 * np.sin(angle)
        field_pt = np.array([[[cx, cy]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
        circle_pts.append(img_pt[0][0])
    circle_pts = np.array(circle_pts, dtype=np.int32)
    cv2.polylines(overlay, [circle_pts], True, (255, 255, 0), 2)

    # Torräume - Magenta
    for goal_x in [0, FIELD_WIDTH]:
        tor_pts = []
        for angle in np.linspace(-np.pi/2, np.pi/2, 30):
            sign = 1 if goal_x == 0 else -1
            cx = goal_x + sign * 2.85 * np.cos(angle)
            cy = 10 + 2.85 * np.sin(angle)
            if (goal_x == 0 and cx >= 0) or (goal_x == FIELD_WIDTH and cx <= FIELD_WIDTH):
                field_pt = np.array([[[cx, cy]]], dtype=np.float32)
                img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
                tor_pts.append(img_pt[0][0])
        if tor_pts:
            tor_pts = np.array(tor_pts, dtype=np.int32)
            cv2.polylines(overlay, [tor_pts], False, (255, 0, 255), 2)

    # Referenzpunkte markieren
    for i, (img_x, img_y, field_x, field_y) in enumerate(POINT_PAIRS):
        cv2.circle(overlay, (int(img_x), int(img_y)), 10, (0, 165, 255), -1)
        cv2.circle(overlay, (int(img_x), int(img_y)), 10, (255, 255, 255), 2)
        cv2.putText(overlay, f"{i+1}", (int(img_x)-5, int(img_y)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlay, f"({field_x},{field_y})", (int(img_x)+15, int(img_y)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return overlay


def main():
    print("=" * 60)
    print("SPIELFELD-OVERLAY GENERATOR")
    print("=" * 60)

    # Bild laden
    print(f"\nLade Bild: {IMAGE_PATH}")
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        print(f"FEHLER: Bild nicht gefunden!")
        return

    h, w = img.shape[:2]
    print(f"Bildgröße: {w} x {h} Pixel")

    # Punkt-Paare
    print(f"\nPunkt-Paare: {len(POINT_PAIRS)}")
    for i, (img_x, img_y, field_x, field_y) in enumerate(POINT_PAIRS):
        print(f"  {i+1}: Bild({img_x}, {img_y}) -> Feld({field_x}, {field_y})m")

    if len(POINT_PAIRS) < 4:
        print("\nFEHLER: Mindestens 4 Punkt-Paare nötig!")
        return

    # Homography berechnen
    image_pts = np.array([(p[0], p[1]) for p in POINT_PAIRS], dtype=np.float32)
    field_pts = np.array([(p[2], p[3]) for p in POINT_PAIRS], dtype=np.float32)

    homography, status = cv2.findHomography(image_pts, field_pts, cv2.RANSAC, 5.0)
    if homography is None:
        print("\nFEHLER: Homography konnte nicht berechnet werden!")
        return

    print("\nHomography Matrix:")
    print(homography)

    # Overlay generieren
    homography_inv = np.linalg.inv(homography)
    overlay = draw_field_overlay(img, homography_inv)

    # Speichern
    cv2.imwrite(str(OUTPUT_PATH), overlay)
    print(f"\n✓ Overlay gespeichert: {OUTPUT_PATH}")
    print("\nÖffne das Bild und prüfe ob das Spielfeld passt!")
    print("Falls nicht: Passe POINT_PAIRS im Script an und führe erneut aus.")


if __name__ == "__main__":
    main()
