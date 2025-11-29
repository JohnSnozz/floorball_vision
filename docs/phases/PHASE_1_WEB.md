# Phase 1: Web-Grundgerüst

## Ziel
Flask-App mit Video-Upload, YouTube-Download und Video-Verwaltung.

## Voraussetzungen
- [x] Phase 0 abgeschlossen
- [x] PostgreSQL läuft
- [x] Redis läuft

---

## Schritt-für-Schritt Plan

### Schritt 1.1: Flask App Factory
**Datei:** `src/web/app.py`

```python
# Flask App mit:
# - SQLAlchemy
# - Celery
# - Blueprints
# - Config aus .env
```

**Test:**
```bash
python -c "from src.web.app import create_app; app = create_app(); print(app.name)"
# Erwartet: src.web.app
```

---

### Schritt 1.2: Konfiguration
**Datei:** `src/web/config.py`

```python
# Config Klassen:
# - BaseConfig
# - DevelopmentConfig
# - ProductionConfig
```

**Test:**
```bash
python -c "from src.web.config import DevelopmentConfig; print(DevelopmentConfig.SQLALCHEMY_DATABASE_URI)"
# Erwartet: postgresql://...
```

---

### Schritt 1.3: SQLAlchemy Models
**Datei:** `src/web/models.py`

Models:
- `Video` (id, title, source_url, file_path, status, etc.)

**Test:**
```bash
python -c "from src.web.models import Video; print(Video.__tablename__)"
# Erwartet: videos
```

---

### Schritt 1.4: Video Routes Blueprint
**Datei:** `src/web/routes/videos.py`

Endpoints:
- `GET /api/videos` - Liste aller Videos
- `GET /api/videos/<id>` - Video Details
- `POST /api/videos/upload` - Datei hochladen
- `POST /api/videos/youtube` - YouTube Download starten
- `GET /api/videos/<id>/status` - Download Status
- `DELETE /api/videos/<id>` - Video löschen

**Test:**
```bash
# Flask starten
python -m src.web.app &
sleep 2

# API Tests
curl http://localhost:5000/api/videos
# Erwartet: []

curl -X POST http://localhost:5000/api/videos/youtube \
     -H "Content-Type: application/json" \
     -d '{"url": "https://youtube.com/watch?v=test"}'
# Erwartet: {"id": "...", "status": "downloading"}
```

---

### Schritt 1.5: Celery Tasks
**Datei:** `src/processing/tasks.py`

Tasks:
- `download_youtube_video(video_id, url)` - YouTube Download
- `extract_video_metadata(video_id)` - Metadaten extrahieren
- `generate_thumbnail(video_id)` - Thumbnail erstellen

**Test:**
```bash
# Celery Worker starten
celery -A src.web.app.celery worker --loglevel=info &
sleep 3

# Task direkt aufrufen
python -c "
from src.processing.tasks import extract_video_metadata
result = extract_video_metadata.delay('test-id')
print(result.status)
"
# Erwartet: PENDING
```

---

### Schritt 1.6: YouTube Downloader
**Datei:** `src/processing/downloader.py`

Funktionen:
- `download_video(url, output_dir)` - Video herunterladen
- `get_video_info(url)` - Metadaten ohne Download
- `DownloadProgress` - Callback für Fortschritt

**Test:**
```bash
python -c "
from src.processing.downloader import get_video_info
info = get_video_info('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
print(info['title'])
"
# Erwartet: Video Titel
```

---

### Schritt 1.7: Base Template
**Datei:** `src/web/templates/base.html`

Features:
- Tailwind CSS (CDN)
- Navigation
- Flash Messages
- Footer

**Test:**
```bash
# Flask starten und Browser öffnen
python -m src.web.app
# http://localhost:5000 - Seite lädt ohne Fehler
```

---

### Schritt 1.8: Index Seite
**Datei:** `src/web/templates/index.html`

Features:
- Video Upload Form
- YouTube URL Form
- Video Liste

**Test:**
```bash
# Browser: http://localhost:5000
# - Upload-Bereich sichtbar
# - YouTube-URL-Eingabe sichtbar
# - Video-Liste (leer) sichtbar
```

---

### Schritt 1.9: JavaScript API Client
**Datei:** `src/web/static/js/api.js`

Funktionen:
- `api.videos.list()`
- `api.videos.get(id)`
- `api.videos.upload(file)`
- `api.videos.downloadYoutube(url)`
- `api.videos.delete(id)`

**Test:**
```bash
# In Browser Console:
api.videos.list().then(console.log)
# Erwartet: []
```

---

### Schritt 1.10: Main JavaScript
**Datei:** `src/web/static/js/app.js`

Features:
- Video Upload Handler
- YouTube Download Handler
- Progress Updates (Polling)
- Video Liste aktualisieren

**Test:**
```bash
# Browser: http://localhost:5000
# 1. Datei hochladen → Video erscheint in Liste
# 2. YouTube URL eingeben → Download startet, Progress sichtbar
```

---

### Schritt 1.11: Video Utils
**Datei:** `src/utils/video_utils.py`

Funktionen:
- `get_video_metadata(path)` - FPS, Dauer, Auflösung
- `extract_frame(path, timestamp)` - Einzelnen Frame extrahieren
- `generate_thumbnail(path, output)` - Thumbnail erstellen

**Test:**
```bash
python -c "
from src.utils.video_utils import get_video_metadata
# Wenn ein Testvideo vorhanden ist:
# meta = get_video_metadata('data/videos/test/original.mp4')
# print(meta)
print('OK')
"
```

---

### Schritt 1.12: File Upload Handler
**Datei:** `src/web/routes/videos.py` (erweitern)

Features:
- Chunked Upload für grosse Dateien
- Progress Tracking
- Validierung (nur Video-Formate)

**Test:**
```bash
# Grosse Datei (>100MB) hochladen
# Progress wird angezeigt
# Video erscheint in Liste
```

---

## Abschluss-Checkliste

- [ ] Flask App startet ohne Fehler
- [ ] Celery Worker startet ohne Fehler
- [ ] Video Upload funktioniert
- [ ] YouTube Download funktioniert
- [ ] Video Liste wird angezeigt
- [ ] Video kann gelöscht werden
- [ ] Thumbnails werden generiert
- [ ] Metadaten werden extrahiert
- [ ] Alle API Endpoints funktionieren

## Finale Tests

```bash
# 1. Alle Services starten
docker-compose up -d
python -m src.web.app &
celery -A src.web.app.celery worker &
sleep 5

# 2. API Tests
curl http://localhost:5000/api/videos
# Erwartet: []

# 3. Upload Test (mit test.mp4)
curl -X POST http://localhost:5000/api/videos/upload \
     -F "file=@test.mp4"
# Erwartet: {"id": "...", "status": "processing"}

# 4. YouTube Test
curl -X POST http://localhost:5000/api/videos/youtube \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.youtube.com/watch?v=SHORT_VIDEO"}'
# Erwartet: {"id": "...", "status": "downloading"}

# 5. Liste prüfen (nach ein paar Sekunden)
curl http://localhost:5000/api/videos
# Erwartet: 2 Videos in Liste

# 6. Browser Test
# http://localhost:5000 - Alles funktioniert

echo "Phase 1 abgeschlossen!"
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 1.5"
2. Git Commit: `git commit -m "Phase 1: Web-Grundgerüst abgeschlossen"`
3. Git Tag: `git tag phase-1-complete`
