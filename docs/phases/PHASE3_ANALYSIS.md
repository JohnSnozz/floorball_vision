# Phase 3: Video-Analyse mit Tracking

## Übersicht

Phase 3 implementiert die taktische Video-Analyse mit:
- YOLO-basierte Spieler-Erkennung
- ByteTrack für konsistentes ID-Tracking
- CLIP-basierte Team-Zuweisung
- Fisheye-korrigierte Koordinaten-Transformation
- Taktische 2D-Visualisierung

## Implementierte Features

### 1. Snippet-basierte Analyse
- Videos werden in kurze Snippets (5-30 Sekunden) unterteilt
- Jedes Snippet wird separat analysiert
- Clips werden automatisch extrahiert

### 2. YOLO Tracking
- Standard YOLOv8 für Personen-Erkennung
- ByteTrack für Frame-übergreifendes Tracking
- Konfigurierbare FPS für Tracking (default: 2 FPS)
- Spielfeld-Polygon-Filter (nur Detektionen im Feld)

### 3. Team-Zuweisung
- CLIP-Modell für Trikot-Farberkennung
- Konfigurierbare Team-Farben
- Mehrheitsvoting über Zeit für stabile Zuweisung
- Optional: Rückennummer-OCR mit EasyOCR

### 4. Koordinaten-Transformation

#### Pixel zu Feld (für Tracking-Punkte)
```
Verzerrtes Video → Fisheye-Entzerrung → Rotation → Homography → Feldkoordinaten
```

#### Feld zu Pixel (für Grid-Overlay)
```
Feldkoordinaten → Inv. Homography → Rotation → Fisheye-Verzerrung → Video-Pixel
```

**Wichtige Parameter:**
- `camera_matrix`: Skaliert von Profil-Auflösung auf Video-Auflösung
- `k1-k4`: Fisheye-Distortion-Koeffizienten
- `zoom_out`: Zoom-Faktor für Entzerrung
- `rotation`: Bildrotation in Grad

### 5. Taktische Ansicht
- Synchronisierter Video-Player mit Overlay
- 2D-Spielfeld-Darstellung
- Spieler-Trails (konfigurierbare Länge)
- Geschwindigkeitssteuerung (0.25x - 2x)
- Grid-Overlay auf Video (optional)

## API Endpoints

### Snippets
- `GET /analysis/api/videos/<id>/snippets` - Alle Snippets
- `POST /analysis/api/videos/<id>/snippets` - Snippets speichern
- `POST /analysis/api/videos/<id>/snippets/<id>/extract` - Clip extrahieren

### Tracking
- `POST /analysis/api/videos/<id>/snippets/<id>/track` - Tracking starten
- `GET /analysis/api/videos/<id>/snippets/<id>/tracking` - Tracking-Daten

### Positionen
- `GET /analysis/api/videos/<id>/snippets/<id>/positions` - Feldpositionen
  - Query: `calibration_id` - Spezifische Kalibrierung verwenden

## Dateien

### Backend
- `src/web/routes/analysis.py` - Alle API-Endpoints und Transformation

### Frontend
- `src/web/templates/analysis/index.html` - Übersicht
- `src/web/templates/analysis/video.html` - Snippet-Auswahl
- `src/web/templates/analysis/tactical.html` - Taktische Ansicht
- `src/web/templates/analysis/review.html` - Tracking-Review

## Datenstruktur

```
data/analysis/<video_id>/
├── snippets.json           # Snippet-Definitionen
├── clips/
│   └── <snippet_id>.mp4    # Extrahierte Clips
├── tracking/
│   └── <snippet_id>.json   # Tracking-Rohdaten
├── positions/
│   └── <snippet_id>.json   # Transformierte Feldpositionen (Cache)
└── boundary.json           # Spielfeld-Polygon (optional)
```

## Koordinatensystem

- Spielfeld: 40m x 20m (Floorball-Standard)
- X-Achse: Längsrichtung (0-40m)
- Y-Achse: Querrichtung (0-20m)
- Ursprung: Linke untere Ecke

## Bekannte Einschränkungen

- Tracking nur auf extrahierten Clips (nicht live)
- Team-Zuweisung benötigt klare Trikot-Farben
- OCR für Rückennummern ist experimentell
- Grid-Overlay Genauigkeit hängt von Kalibrierung ab
