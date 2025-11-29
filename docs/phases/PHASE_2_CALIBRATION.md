# Phase 2: Kamera-Kalibrierung

## Ziel
Spielfeld im Video einzeichnen mit Fisheye-Korrektur und Homography-Transformation.

## Voraussetzungen
- [x] Phase 1.5 abgeschlossen
- [x] Mindestens ein Video vorhanden

---

## Schritt-für-Schritt Plan

### Schritt 2.1: Calibration Routes
**Datei:** `src/web/routes/calibration.py`

Endpoints:
- `GET /api/videos/<id>/calibration` - Kalibrierung abrufen
- `POST /api/videos/<id>/calibration/screenshots` - Screenshots generieren
- `POST /api/videos/<id>/calibration` - Kalibrierung speichern
- `POST /api/videos/<id>/calibration/test` - Test-Overlay generieren

### Schritt 2.2: Calibration Processing
**Datei:** `src/processing/calibration.py`

Funktionen:
- `extract_calibration_screenshots(video_path, count)`
- `calculate_homography(src_points, dst_points)`
- `undistort_fisheye(frame, polygon_points)`
- `transform_point(point, homography_matrix)`

### Schritt 2.3: Canvas Drawing JavaScript
**Datei:** `src/web/static/js/calibration.js`

Features:
- Polygon-Punkte setzen (Fisheye-Bande)
- Spielfeld-Punkte setzen
- Punkt-Korrespondenz anzeigen
- Undo/Redo

### Schritt 2.4: Calibration Templates
**Dateien:**
- `src/web/templates/calibration/setup.html`
- `src/web/templates/calibration/fisheye.html`
- `src/web/templates/calibration/field_points.html`

---

## Tests

```bash
# 1. Screenshots generieren
curl -X POST http://localhost:5000/api/videos/{id}/calibration/screenshots \
     -d '{"count": 5}'
# Screenshots werden erstellt

# 2. Kalibrierung speichern
curl -X POST http://localhost:5000/api/videos/{id}/calibration \
     -d '{"field_points_image": [...], "field_points_tactical": [...]}'
# Kalibrierung in DB

# 3. Test-Overlay
curl http://localhost:5000/api/videos/{id}/calibration/test
# Bild mit Grid-Overlay
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 3"
2. Git Commit: `git commit -m "Phase 2: Kamera-Kalibrierung abgeschlossen"`
3. Git Tag: `git tag phase-2-complete`
