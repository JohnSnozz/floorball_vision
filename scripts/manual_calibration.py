#!/usr/bin/env python3
"""
Manuelle Kalibrierung - Direkt und einfach.

Zeigt das entzerrte Bild an, du klickst Punkte, es zeichnet das Spielfeld-Overlay.
"""
import cv2
import numpy as np
from pathlib import Path

# Pfade
VIDEO_ID = "340d8ea8-00dc-4415-95da-ae18d54daf6b"
PROJECT_ROOT = Path(__file__).parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration" / VIDEO_ID

# Bild laden
IMAGE_PATH = CALIBRATION_DIR / "undistorted_frame.jpg"

# Spielfeld: 40m x 20m
FIELD_WIDTH = 40.0
FIELD_HEIGHT = 20.0

# Gespeicherte Punkt-Paare
point_pairs = []
current_image_point = None

# Fenster
WINDOW_NAME = "Kalibrierung - Klicke Punkte"

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
        for x in np.linspace(0, FIELD_WIDTH, 100):
            field_pt = np.array([[[x, y]]], dtype=np.float32)
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

    return overlay


def redraw():
    """Zeichnet das Bild mit allen Punkten neu."""
    global img_display

    img_display = img_original.copy()

    # Punkt-Paare zeichnen
    for i, pair in enumerate(point_pairs):
        x, y = int(pair['image'][0]), int(pair['image'][1])
        cv2.circle(img_display, (x, y), 10, (0, 165, 255), -1)
        cv2.circle(img_display, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(img_display, str(i+1), (x-6, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Feld-Koordinaten daneben
        fx, fy = pair['field']
        cv2.putText(img_display, f"({fx:.1f},{fy:.1f})", (x+15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Aktueller Punkt (noch kein Paar)
    if current_image_point:
        x, y = current_image_point
        cv2.circle(img_display, (x, y), 12, (0, 255, 255), 3)

    # Wenn genug Punkte, Overlay zeichnen
    if len(point_pairs) >= 4:
        image_pts = np.array([p['image'] for p in point_pairs], dtype=np.float32)
        field_pts = np.array([p['field'] for p in point_pairs], dtype=np.float32)

        homography, _ = cv2.findHomography(image_pts, field_pts, cv2.RANSAC, 5.0)
        if homography is not None:
            homography_inv = np.linalg.inv(homography)
            img_display = draw_field_overlay(img_display, homography_inv)

    # Anleitung
    h, w = img_display.shape[:2]
    cv2.rectangle(img_display, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.putText(img_display, f"Punkte: {len(point_pairs)}/4+  |  Linksklick: Punkt setzen  |  Rechtsklick: Letzten entfernen",
                (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_display, "Nach Klick: Gib Feld-Koordinaten ein (z.B. '20 10' fuer Mittelpunkt, '0 0' fuer Ecke unten-links)",
                (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, img_display)


def mouse_callback(event, x, y, flags, param):
    global current_image_point, point_pairs

    if event == cv2.EVENT_LBUTTONDOWN:
        current_image_point = (x, y)
        print(f"\nBild-Punkt gesetzt: ({x}, {y})")
        print("Gib jetzt die Feld-Koordinaten ein (z.B. '20 10' für Mittelpunkt):")
        redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if point_pairs:
            removed = point_pairs.pop()
            print(f"Punkt entfernt: {removed}")
            redraw()


def main():
    global img_original, img_display, current_image_point, point_pairs

    print("=" * 60)
    print("MANUELLE KALIBRIERUNG")
    print("=" * 60)
    print(f"\nBild: {IMAGE_PATH}")
    print("\nSpielfeld-Koordinaten (Meter):")
    print("  Ecken: (0,0)=unten-links, (40,0)=unten-rechts")
    print("         (0,20)=oben-links, (40,20)=oben-rechts")
    print("  Mitte: (20,10)")
    print("  Tore:  (0,10)=links, (40,10)=rechts")
    print("\nBedienung:")
    print("  - Linksklick: Punkt im Bild setzen")
    print("  - Dann Koordinaten im Terminal eingeben")
    print("  - Rechtsklick: Letzten Punkt entfernen")
    print("  - 's': Speichern und beenden")
    print("  - 'q': Beenden ohne speichern")
    print("=" * 60)

    # Bild laden
    img_original = cv2.imread(str(IMAGE_PATH))
    if img_original is None:
        print(f"FEHLER: Bild nicht gefunden: {IMAGE_PATH}")
        return

    print(f"\nBild geladen: {img_original.shape[1]}x{img_original.shape[0]}")

    # Fenster erstellen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 800)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    redraw()

    while True:
        key = cv2.waitKey(100) & 0xFF

        # Prüfe ob Koordinaten-Eingabe nötig
        if current_image_point is not None:
            try:
                coords = input().strip()
                if coords:
                    parts = coords.replace(',', ' ').split()
                    if len(parts) >= 2:
                        fx, fy = float(parts[0]), float(parts[1])
                        point_pairs.append({
                            'image': current_image_point,
                            'field': (fx, fy)
                        })
                        print(f"Punkt-Paar hinzugefügt: Bild({current_image_point[0]}, {current_image_point[1]}) -> Feld({fx}, {fy})")
                        current_image_point = None
                        redraw()
            except EOFError:
                pass

        if key == ord('s'):
            if len(point_pairs) >= 4:
                # Speichern
                image_pts = np.array([p['image'] for p in point_pairs], dtype=np.float32)
                field_pts = np.array([p['field'] for p in point_pairs], dtype=np.float32)
                homography, _ = cv2.findHomography(image_pts, field_pts, cv2.RANSAC, 5.0)

                # Overlay speichern
                homography_inv = np.linalg.inv(homography)
                overlay = draw_field_overlay(img_original, homography_inv)
                output_path = CALIBRATION_DIR / "manual_overlay.jpg"
                cv2.imwrite(str(output_path), overlay)
                print(f"\nOverlay gespeichert: {output_path}")

                # Kalibrierung speichern
                import json
                calib_data = {
                    'point_pairs': point_pairs,
                    'homography': homography.tolist(),
                    'image_size': [img_original.shape[1], img_original.shape[0]]
                }
                calib_path = CALIBRATION_DIR / "manual_calibration.json"
                with open(calib_path, 'w') as f:
                    json.dump(calib_data, f, indent=2)
                print(f"Kalibrierung gespeichert: {calib_path}")
                break
            else:
                print("Mindestens 4 Punkte nötig!")

        elif key == ord('q'):
            print("Abgebrochen.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
