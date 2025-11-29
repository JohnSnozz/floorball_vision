# Phase 3: YOLO Tracking

## Ziel
Spieler und Ball erkennen, tracken und Teams zuweisen.

## Voraussetzungen
- [x] Phase 2 abgeschlossen
- [x] Trainiertes YOLO Modell vorhanden

---

## Schritt-für-Schritt Plan

### Schritt 3.1: Base Tracker
**Datei:** `src/trackers/base_tracker.py`

Abstrakte Klasse für alle Tracker.

### Schritt 3.2: Player Tracker
**Datei:** `src/trackers/player_tracker.py`

- YOLO Detection
- ByteTrack für konsistente IDs
- Batch Processing

### Schritt 3.3: Ball Tracker
**Datei:** `src/trackers/ball_tracker.py`

- YOLO Detection
- Interpolation für fehlende Frames
- Outlier-Filterung

### Schritt 3.4: Team Assigner
**Datei:** `src/analysis/team_assigner.py`

- CLIP Modell für Trikot-Erkennung
- Team-Caching

### Schritt 3.5: Position Mapper
**Datei:** `src/analysis/position_mapper.py`

- Pixel zu Meter Transformation
- Kalibrierung anwenden

### Schritt 3.6: Detection Test Endpoint
**Datei:** `src/web/routes/analysis.py`

- `POST /api/analysis/test` - Einzelnen Frame testen

---

## Tests

```bash
# 1. Single Frame Detection
curl -X POST http://localhost:5000/api/analysis/test \
     -d '{"video_id": "...", "frame_number": 100}'
# Bounding Boxes im Response

# 2. Tracking Test (10 Frames)
curl -X POST http://localhost:5000/api/analysis/test \
     -d '{"video_id": "...", "start_frame": 100, "end_frame": 110}'
# Konsistente Track IDs
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 4"
2. Git Commit: `git commit -m "Phase 3: YOLO Tracking abgeschlossen"`
3. Git Tag: `git tag phase-3-complete`
