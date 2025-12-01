# Changelog

## [Unreleased]

### Phase 3: Video-Analyse mit Tracking (2025-12-01)

#### Neue Features
- **Snippet-basierte Analyse**
  - Video-Snippets definieren (5-30 Sekunden)
  - Automatische Clip-Extraktion mit FFmpeg
  - Mehrere Snippets pro Video möglich

- **YOLO Tracking**
  - Personen-Erkennung mit YOLOv8
  - ByteTrack für konsistentes ID-Tracking
  - Spielfeld-Polygon-Filter für relevante Detektionen

- **Team-Zuweisung**
  - CLIP-basierte Trikot-Farberkennung
  - Konfigurierbare Team-Farben
  - Mehrheitsvoting für stabile Zuweisung

- **Taktische Ansicht**
  - Synchronisierter Video-Player mit Tracking-Overlay
  - 2D-Spielfeld-Darstellung mit Spielerpositionen
  - Konfigurierbare Trails
  - Geschwindigkeitssteuerung (0.25x - 2x)
  - Spielfeld-Grid-Overlay auf Video

- **Kalibrierungs-Integration**
  - Auswahl der Kalibrierung im Analyse-Schritt
  - Korrekte Fisheye-Transformation für Grid-Projektion
  - Präzise Pixel-zu-Feld-Koordinaten-Transformation

#### Verbesserungen
- **Kalibrierung Review-Modus**
  - Bestehende Kalibrierungen können Schritt für Schritt angezeigt werden
  - Alle 5 Schritte navigierbar ohne Neuberechnung

- **Fisheye-Transformation**
  - Grid-Overlay folgt jetzt korrekt der Fisheye-Krümmung
  - Segmentierte Linien (0.5m) für glatte Kurven
  - Korrekte Rotation und Zoom-Behandlung

#### Bugfixes
- TypeError bei UUID-Slicing im Template behoben
- Koordinatentransformation für Tracking-Punkte korrigiert
- Rotation mit korrektem Vorzeichen für Pixel→Feld-Transformation

### Phase 2: Kamera-Kalibrierung (2025-11-30)

#### Neue Features
- 5-Schritt Kalibrierungs-Wizard
- Fisheye-Entzerrung mit Lens-Profilen
- Custom-Modus für manuelle k1-k4 Parameter
- Punkt-Paar-basierte Homography-Berechnung
- Schlusskontrolle auf Original-Bild

### Phase 1: Web-Grundgerüst (2025-11-29)

#### Neue Features
- Flask App mit SQLAlchemy und Celery
- Video-Upload und YouTube-Download
- REST API für Videos
- Metadaten-Extraktion und Thumbnails

### Phase 0: Setup & Infrastruktur (2025-11-29)

#### Initial Setup
- PostgreSQL Datenbank
- Redis für Celery
- Alembic Migrations
- Projekt-Struktur
