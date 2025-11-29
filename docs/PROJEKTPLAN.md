# Floorball Vision - Projektplan

## Übersicht

Ein webbasiertes System zur Analyse von Floorball-Videos mit automatischer Spieler-Erkennung, Positions-Tracking und integriertem Labeling-Workflow.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLOORBALL VISION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    TRAINING PIPELINE (Label Studio)                   │  │
│  │  Screenshots ──▶ Label Studio ──▶ Export ──▶ YOLO Training ──▶ Model │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                       │                                     │
│                                       ▼                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Phase 1   │───▶│   Phase 2   │───▶│   Phase 3   │───▶│   Phase 4   │ │
│  │  Web-Setup  │    │ Kalibrierung│    │  Tracking   │    │   Preview   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│         │                                                        │         │
│         ▼                                                        ▼         │
│  ┌─────────────┐                                          ┌─────────────┐ │
│  │   Phase 5   │─────────────────────────────────────────▶│   Phase 6   │ │
│  │  Full Run   │                                          │   Export    │ │
│  └─────────────┘                                          └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technologie-Stack

### Backend (Python)
| Komponente | Technologie | Grund |
|------------|-------------|-------|
| Web Framework | **Flask** | Leichtgewichtig, Python-native |
| Task Queue | **Celery + Redis** | Async Jobs für Video-Processing |
| Datenbank | **PostgreSQL** | Robust, JSON-Support, skalierbar |
| ORM | **SQLAlchemy** | Python-Standard, Migrations |
| Video Processing | **OpenCV** | Standard, effizient |
| Object Detection | **Ultralytics YOLO** | State-of-the-art, GPU-optimiert |
| Tracking | **ByteTrack (supervision)** | Bewährt im Basketball-Projekt |
| Labeling | **Label Studio** (extern) | Open Source, YOLO-Export |

### Frontend (KEIN Node.js/npm!)
| Komponente | Technologie | Grund |
|------------|-------------|-------|
| UI | **Vanilla JavaScript (ES6+)** | Kein Build-Step, direkt, schnell |
| Styling | **Tailwind CSS (CDN)** | Utility-first, kein Build nötig |
| Video Player | **HTML5 Video + Custom Controls** | Volle Kontrolle, leichtgewichtig |
| Canvas Drawing | **Native Canvas API** | Kein Framework-Overhead |
| HTTP Requests | **Fetch API** | Modern, native |

> **WICHTIG:** Dieses Projekt verwendet bewusst **KEIN Node.js, npm, webpack, vite oder andere JS-Build-Tools**.
> - Vanilla JavaScript direkt im Browser
> - CSS via CDN (Tailwind)
> - Keine `package.json`, keine `node_modules`
> - Grund: Sicherheit (npm hat regelmässig Vulnerabilities), Einfachheit, weniger Abhängigkeiten

### Infrastruktur
| Komponente | Entwicklung | Produktion (AWS) |
|------------|-------------|------------------|
| Datenbank | PostgreSQL lokal | RDS PostgreSQL |
| Redis | Redis lokal | ElastiCache |
| Storage | Lokales Filesystem | S3 |
| Compute | Lokale Maschine | EC2 Spot (GPU) |

---

## Label Studio Integration

### Architektur

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LABEL STUDIO WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FLOORBALL VISION                         LABEL STUDIO (Port 8080)      │
│  ┌────────────────────┐                   ┌────────────────────┐        │
│  │                    │   1. Screenshots  │                    │        │
│  │  Screenshots       │ ───────────────▶  │  Project           │        │
│  │  generieren        │   (API Upload)    │  "floorball_v1"    │        │
│  │                    │                   │                    │        │
│  └────────────────────┘                   └────────────────────┘        │
│                                                    │                     │
│                                                    │ 2. Manuelles        │
│                                                    │    Labeling         │
│                                                    ▼                     │
│  ┌────────────────────┐                   ┌────────────────────┐        │
│  │                    │   3. Webhook      │                    │        │
│  │  Training          │ ◀───────────────  │  Annotations       │        │
│  │  Pipeline          │   oder Polling    │  (YOLO Format)     │        │
│  │                    │                   │                    │        │
│  └────────────────────┘                   └────────────────────┘        │
│           │                                                              │
│           │ 4. Neues Modell                                             │
│           ▼                                                              │
│  ┌────────────────────┐                                                 │
│  │  models/           │                                                 │
│  │  └─ yolo_v2.pt     │                                                 │
│  └────────────────────┘                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Label Studio API Integration

```python
# src/labeling/label_studio_client.py

class LabelStudioClient:
    """Client für Label Studio API Integration."""

    def __init__(self, url: str = "http://localhost:8080", api_key: str = None):
        self.url = url
        self.api_key = api_key or os.getenv("LABEL_STUDIO_API_KEY")

    def create_project(self, name: str, label_config: str) -> dict:
        """Erstellt ein neues Labeling-Projekt."""
        pass

    def upload_images(self, project_id: int, image_paths: list) -> dict:
        """Lädt Bilder in ein Projekt hoch."""
        pass

    def get_annotations(self, project_id: int) -> list:
        """Holt alle Annotations eines Projekts."""
        pass

    def export_yolo(self, project_id: int, output_dir: str) -> str:
        """Exportiert Annotations im YOLO-Format."""
        pass

    def get_project_stats(self, project_id: int) -> dict:
        """Gibt Labeling-Fortschritt zurück."""
        pass
```

### Automatisches Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRAINING VERWALTUNG                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LABEL STUDIO PROJEKTE                                                  │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Projekt: floorball_v1                                           │   │
│  │ Status: 450/500 Bilder gelabelt (90%)                          │   │
│  │ Letzte Aktivität: vor 2 Stunden                                │   │
│  │                                                                 │   │
│  │ [Label Studio öffnen]  [Annotations exportieren]               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  TRAININGS-HISTORIE                                                     │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                          │
│  │ Version │ Datum      │ Bilder │ mAP50  │ Status    │ Aktionen │     │
│  │─────────│────────────│────────│────────│───────────│──────────│     │
│  │ v3      │ 2024-01-15 │ 450    │ 0.847  │ ✓ Aktiv   │ [Nutzen] │     │
│  │ v2      │ 2024-01-10 │ 300    │ 0.782  │ Archiv    │ [Laden]  │     │
│  │ v1      │ 2024-01-05 │ 150    │ 0.654  │ Archiv    │ [Laden]  │     │
│                                                                          │
│  NEUES TRAINING STARTEN                                                 │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                          │
│  Quelle: [Label Studio Projekt: floorball_v1 ▼]                        │
│  Basis-Modell: [yolov8n.pt ▼]                                          │
│  Epochs: [100]                                                          │
│  Batch Size: [16]                                                       │
│                                                                          │
│  Geschätzte Dauer: ~2 Stunden (GPU)                                    │
│                                                                          │
│  [Training starten]                                                     │
│                                                                          │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                          │
│  AKTUELLES TRAINING                                                     │
│                                                                          │
│  ████████████████░░░░░░░░░░░░░░░░  Epoch 52/100                        │
│                                                                          │
│  Loss: 0.0234  │  mAP50: 0.823  │  mAP50-95: 0.712                     │
│                                                                          │
│  [Pause] [Abbrechen]                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Datei-Struktur (Final)

```
floorball_vision/
├── src/
│   ├── web/                          # Flask Application
│   │   ├── __init__.py
│   │   ├── app.py                   # Flask App Factory
│   │   ├── config.py                # Konfiguration (DB, Redis, etc.)
│   │   ├── models.py                # SQLAlchemy Models
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── videos.py            # Video Upload/Download
│   │   │   ├── calibration.py       # Kamera-Kalibrierung
│   │   │   ├── analysis.py          # Tracking & Analyse
│   │   │   ├── training.py          # Model Training Management
│   │   │   ├── labeling.py          # Label Studio Integration
│   │   │   └── export.py            # Daten-Export
│   │   ├── templates/
│   │   │   ├── base.html
│   │   │   ├── index.html
│   │   │   ├── videos/
│   │   │   ├── calibration/
│   │   │   ├── analysis/
│   │   │   ├── training/
│   │   │   │   └── dashboard.html
│   │   │   └── labeling/
│   │   │       └── projects.html
│   │   └── static/
│   │       ├── css/
│   │       │   └── main.css
│   │       ├── js/
│   │       │   ├── app.js           # Haupt-JavaScript
│   │       │   ├── calibration.js   # Canvas-Zeichnung
│   │       │   ├── video-player.js  # Custom Video Controls
│   │       │   ├── snippets.js      # Snippet-Auswahl
│   │       │   └── api.js           # API-Client
│   │       └── images/
│   │           └── field.svg
│   │
│   ├── labeling/                    # Label Studio Integration
│   │   ├── __init__.py
│   │   ├── client.py                # Label Studio API Client
│   │   ├── export.py                # YOLO Export Handler
│   │   └── sync.py                  # Projekt-Synchronisation
│   │
│   ├── processing/                  # Video & ML Processing
│   │   ├── __init__.py
│   │   ├── downloader.py            # YouTube Download
│   │   ├── frame_extractor.py       # Frame Sampling
│   │   ├── calibration.py           # Homography & Fisheye
│   │   └── tasks.py                 # Celery Tasks
│   │
│   ├── training/                    # Model Training
│   │   ├── __init__.py
│   │   ├── trainer.py               # YOLO Training Wrapper
│   │   ├── dataset.py               # Dataset Preparation
│   │   └── evaluate.py              # Model Evaluation
│   │
│   ├── trackers/                    # Object Tracking
│   │   ├── __init__.py
│   │   ├── player_tracker.py
│   │   ├── ball_tracker.py
│   │   └── base_tracker.py
│   │
│   ├── analysis/                    # Spielanalyse
│   │   ├── __init__.py
│   │   ├── team_assigner.py
│   │   ├── jersey_reader.py
│   │   ├── position_mapper.py
│   │   └── possession.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py
│       ├── bbox_utils.py
│       └── db_utils.py
│
├── data/                            # Daten (gitignored)
│   ├── videos/
│   │   └── {video_id}/
│   │       ├── original.mp4
│   │       ├── metadata.json
│   │       └── thumbnails/
│   ├── frames/
│   │   └── {video_id}/
│   ├── labeling/                    # Label Studio Daten
│   │   ├── exports/                 # YOLO Exports
│   │   │   └── {project_id}/
│   │   │       ├── images/
│   │   │       ├── labels/
│   │   │       └── data.yaml
│   │   └── uploads/                 # Hochgeladene Bilder
│   ├── training/                    # Training Datasets
│   │   └── {training_id}/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── cache/
│   └── exports/
│
├── models/                          # ML Models (gitignored)
│   ├── base/                        # Basis-Modelle (Download)
│   │   ├── yolov8n.pt
│   │   └── yolov8s.pt
│   ├── trained/                     # Trainierte Modelle
│   │   ├── v1_20240105/
│   │   │   ├── weights/
│   │   │   │   └── best.pt
│   │   │   └── metrics.json
│   │   └── v2_20240110/
│   └── active/                      # Aktuell verwendetes Modell
│       └── model.pt -> ../trained/v3_20240115/weights/best.pt
│
├── configs/
│   ├── classes.yaml
│   ├── field_dimensions.yaml
│   ├── label_studio.yaml            # Label Studio Konfiguration
│   └── training_defaults.yaml
│
├── migrations/                      # Alembic DB Migrations
├── tests/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml           # App + PostgreSQL + Redis
│   └── docker-compose.dev.yml
│
├── scripts/
│   ├── setup_db.py                  # Datenbank initialisieren
│   ├── setup_label_studio.py        # Label Studio Projekt erstellen
│   └── train_model.py               # CLI für Training
│
├── requirements.txt
├── setup.py
├── PROJECT.md
└── docs/
    ├── PROJEKTPLAN.md
    └── API.md
```

---

## Datenbank-Schema (PostgreSQL)

```sql
-- Videos
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    source_url TEXT,
    file_path TEXT NOT NULL,
    duration_seconds REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Kamera-Kalibrierungen
CREATE TABLE calibrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    name VARCHAR(100),
    fisheye_enabled BOOLEAN DEFAULT FALSE,
    fisheye_params JSONB,
    field_points_image JSONB,
    field_points_tactical JSONB,
    homography_matrix JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Label Studio Projekte
CREATE TABLE labeling_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label_studio_id INTEGER UNIQUE,      -- ID in Label Studio
    name VARCHAR(255) NOT NULL,
    description TEXT,
    total_images INTEGER DEFAULT 0,
    labeled_images INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Training Runs
CREATE TABLE training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    labeling_project_id UUID REFERENCES labeling_projects(id),
    version VARCHAR(50) NOT NULL,        -- z.B. "v3"
    base_model VARCHAR(100),             -- z.B. "yolov8n.pt"
    epochs INTEGER,
    batch_size INTEGER,
    image_size INTEGER,

    -- Resultate
    images_train INTEGER,
    images_val INTEGER,
    map50 REAL,
    map50_95 REAL,
    precision_val REAL,
    recall_val REAL,

    -- Pfade
    dataset_path TEXT,
    weights_path TEXT,

    status VARCHAR(50) DEFAULT 'pending',
    progress REAL DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Aktives Modell (nur eine Zeile)
CREATE TABLE active_model (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    training_run_id UUID REFERENCES training_runs(id),
    activated_at TIMESTAMP DEFAULT NOW()
);

-- Analyse-Jobs
CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    calibration_id UUID REFERENCES calibrations(id),
    training_run_id UUID REFERENCES training_runs(id),

    start_time_seconds REAL,
    end_time_seconds REAL,
    sample_rate INTEGER DEFAULT 5,

    status VARCHAR(50) DEFAULT 'pending',
    progress REAL DEFAULT 0,
    total_frames INTEGER,
    processed_frames INTEGER DEFAULT 0,

    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Tracking-Daten (partitioniert für Performance)
CREATE TABLE player_positions (
    id BIGSERIAL,
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER,

    track_id INTEGER,                    -- ByteTrack ID
    player_class VARCHAR(20),            -- 'player', 'goalkeeper', 'ref'
    team_id SMALLINT,
    jersey_number VARCHAR(10),

    -- Bild-Koordinaten
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    confidence REAL,

    -- Spielfeld-Koordinaten (Meter)
    field_x REAL,
    field_y REAL,

    PRIMARY KEY (job_id, frame_number, track_id)
) PARTITION BY LIST (job_id);

-- Ball-Positionen
CREATE TABLE ball_positions (
    id BIGSERIAL,
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER,

    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    confidence REAL,

    field_x REAL,
    field_y REAL,
    possession_track_id INTEGER,

    PRIMARY KEY (job_id, frame_number)
) PARTITION BY LIST (job_id);

-- Indizes
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_training_runs_status ON training_runs(status);
CREATE INDEX idx_analysis_jobs_video ON analysis_jobs(video_id);
CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
```

---

## Phase 0 (NEU): Setup & Infrastruktur

**Ziel:** Entwicklungsumgebung mit PostgreSQL, Redis und Label Studio einrichten

```
┌─────────────────────────────────────────────────────────────────────┐
│  SETUP CHECKLIST                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [✓] PostgreSQL 16 installiert                                      │
│  [ ] Datenbank "floorball_vision" erstellt                          │
│  [ ] Redis installiert und gestartet                                │
│  [ ] Label Studio läuft auf Port 8080                               │
│  [ ] Label Studio API Key generiert                                 │
│  [ ] .env Datei konfiguriert                                        │
│                                                                      │
│  SERVICES                                                           │
│  ───────────────────────────────────────────────────────────────── │
│                                                                      │
│  PostgreSQL:    [●] Running    localhost:5432                       │
│  Redis:         [○] Stopped    localhost:6379                       │
│  Label Studio:  [○] Stopped    localhost:8080                       │
│  Flask App:     [○] Stopped    localhost:5000                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Tasks:**
- [ ] PostgreSQL Datenbank erstellen
- [ ] Redis installieren/starten
- [ ] Label Studio Projekt für Floorball erstellen
- [ ] `.env` Datei mit Credentials
- [ ] `docker-compose.yml` für einfaches Setup
- [ ] DB Schema mit Alembic Migrations

**Dein Kontrollpunkt:**
```bash
# Services starten
docker-compose up -d  # PostgreSQL + Redis

# Label Studio starten (separates Terminal)
label-studio start --port 8080

# Datenbank prüfen
psql -h localhost -U postgres -d floorball_vision -c "\\dt"

# Flask App starten
python -m src.web.app
# Browser: http://localhost:5000
```

---

## Phase 1: Web-Grundgerüst

**Ziel:** Flask-App mit Video-Upload/Download und PostgreSQL

```
┌─────────────────────────────────────────────────────────────────────┐
│  FLOORBALL VISION                                      [Training ▼] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  VIDEO HINZUFÜGEN                                            │  │
│  │                                                              │  │
│  │  ○ YouTube URL                                              │  │
│  │    [https://youtube.com/watch?v=...           ] [Laden]     │  │
│  │                                                              │  │
│  │  ○ Datei hochladen                                          │  │
│  │    [Datei auswählen...                        ] [Upload]    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  MEINE VIDEOS                                          [+ Neu]      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ ┌──────┐                                                       ││
│  │ │      │  Jets vs Sarnen SF2                                   ││
│  │ │ 🎬   │  45:23 min │ 1080p │ 30fps                           ││
│  │ │      │  Status: ✓ Bereit                                    ││
│  │ └──────┘                                                       ││
│  │                                                                ││
│  │  [⚙️ Kalibrieren]  [▶️ Analysieren]  [📊 Ergebnisse]  [🗑️]   ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ ┌──────┐                                                       ││
│  │ │      │  ULA vs Jets QF1                                      ││
│  │ │ ⏳   │  Wird heruntergeladen...                              ││
│  │ │      │  ████████████░░░░░░░░░░░░░░░░  45%                   ││
│  │ └──────┘                                                       ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Tasks:**
- [ ] Flask App Factory mit Blueprints
- [ ] SQLAlchemy Models (PostgreSQL)
- [ ] Alembic Migrations
- [ ] YouTube Download (Celery Task)
- [ ] File Upload mit Progress
- [ ] Video-Liste mit Status-Updates
- [ ] Basis-Templates (Vanilla JS)
- [ ] Thumbnail-Generierung

**Dein Kontrollpunkt:**
```bash
# Flask starten
python -m src.web.app

# Celery Worker starten (separates Terminal)
celery -A src.web.app.celery worker --loglevel=info

# Test im Browser
# 1. http://localhost:5000 öffnen
# 2. YouTube URL eingeben
# 3. Download-Fortschritt beobachten
# 4. Video erscheint in Liste
```

---

## Phase 1.5 (NEU): Label Studio Integration

**Ziel:** Screenshots zu Label Studio pushen, Modell-Training auslösen

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING & LABELING                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LABEL STUDIO PROJEKTE                              [+ Neues Projekt]│
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  📁 floorball_main                                         │   │
│  │  ──────────────────────────────────────────────────────── │   │
│  │  Bilder: 523 │ Gelabelt: 487 (93%) │ Letzte: vor 2h       │   │
│  │                                                             │   │
│  │  ████████████████████████████░░░  93%                      │   │
│  │                                                             │   │
│  │  [🔗 In Label Studio öffnen]  [📤 Export]  [🏋️ Training]  │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  SCREENSHOTS HOCHLADEN                                              │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  Quelle: [Video auswählen... ▼]                                    │
│  Anzahl: [50] Screenshots (zufällig verteilt)                      │
│  Ziel-Projekt: [floorball_main ▼]                                  │
│                                                                      │
│  [Screenshots generieren und hochladen]                             │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  TRAINIERTE MODELLE                                                 │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  │ Version │ Datum      │ Bilder │ mAP50 │ Status        │         │
│  │─────────│────────────│────────│───────│───────────────│         │
│  │ v3      │ 15.01.2024 │ 450    │ 0.847 │ ✓ AKTIV      │         │
│  │ v2      │ 10.01.2024 │ 300    │ 0.782 │ archiviert   │         │
│  │ v1      │ 05.01.2024 │ 150    │ 0.654 │ archiviert   │         │
│                                                                      │
│  [Neues Training starten]                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Tasks:**
- [ ] Label Studio API Client
- [ ] Screenshot-Upload zu Label Studio
- [ ] Projekt-Status Synchronisation
- [ ] YOLO Export aus Label Studio
- [ ] Training starten (Celery Task)
- [ ] Modell-Versionierung
- [ ] Aktives Modell wechseln

**Dein Kontrollpunkt:**
```bash
# Label Studio öffnen
# http://localhost:8080

# In Flask App:
# 1. Screenshots aus Video generieren
# 2. Zu Label Studio hochladen
# 3. In Label Studio labeln
# 4. Export + Training starten
# 5. Neues Modell aktivieren
```

---

## Phasen 2-6

*(bleiben wie im vorherigen Plan, mit Anpassungen für PostgreSQL)*

---

## Konfigurationsdateien

### .env (Beispiel)
```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/floorball_vision

# Redis
REDIS_URL=redis://localhost:6379/0

# Label Studio
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your-api-key-here

# Flask
FLASK_SECRET_KEY=your-secret-key
FLASK_ENV=development

# Paths
DATA_DIR=/home/jonas/floorball_vision/data
MODELS_DIR=/home/jonas/floorball_vision/models
```

### configs/label_studio.yaml
```yaml
# Label Studio Konfiguration für Floorball

# Label Interface (XML)
label_config: |
  <View>
    <Image name="image" value="$image"/>
    <RectangleLabels name="label" toName="image">
      <Label value="player" background="#00ff00"/>
      <Label value="goalkeeper" background="#ff0000"/>
      <Label value="ref" background="#ffff00"/>
      <Label value="ball" background="#0000ff"/>
      <Label value="goal" background="#ff00ff"/>
    </RectangleLabels>
  </View>

# Klassen-Mapping zu YOLO
class_mapping:
  player: 0
  goalkeeper: 1
  ref: 2
  ball: 3
  goal: 4

# Export-Einstellungen
export:
  format: YOLO
  include_images: true
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: floorball_vision
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Label Studio läuft separat (nicht in Docker)
  # weil es bereits installiert ist

volumes:
  postgres_data:
```

---

## Nächster Schritt

Soll ich mit **Phase 0 (Setup)** beginnen?

1. PostgreSQL Datenbank erstellen
2. docker-compose.yml für PostgreSQL + Redis
3. Alembic Migrations Setup
4. `.env` Template
5. Label Studio Projekt-Setup Script

Danach kannst du mit `docker-compose up -d` und `label-studio start` alles starten.
