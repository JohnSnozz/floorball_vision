# Floorball Vision

Computer Vision System zur Analyse von Floorball-Spielen mittels YOLO Object Detection.

## Projektziel

Ein End-to-End System zur automatischen Analyse von Floorball-Spielvideos:

1. **Datenakquise** - YouTube-Videos herunterladen und Screenshots extrahieren
2. **Labeling** - Screenshots mit Label Studio annotieren
3. **Training** - YOLO-Modell trainieren
4. **Analyse** - Videos analysieren und Spielerdaten extrahieren
5. **Visualisierung** - Web-Dashboard für Statistiken

## Geplante Features

### Phase 1: Grundlagen (aktuell)
- [x] YouTube Video Download Pipeline
- [x] Screenshot-Extraktion und Batch-Erstellung
- [x] YOLO Training Setup
- [ ] Projekt-Restrukturierung

### Phase 2: Detection & Tracking
- [ ] Spieler-Detection optimieren
- [ ] Ball-Detection verbessern
- [ ] Player Tracking (ByteTrack)
- [ ] Ball Tracking

### Phase 3: Spielanalyse
- [ ] Team-Zuweisung (Trikotfarben)
- [ ] Ballbesitz-Erkennung
- [ ] Pass-Erkennung
- [ ] Spielerkoordinaten extrahieren

### Phase 4: Visualisierung & Deployment
- [ ] Taktische Ansicht (Top-Down View)
- [ ] Flask Web-Interface
- [ ] AWS Deployment
- [ ] API für Daten-Upload

## Erkannte Klassen (9)

| ID | Klasse | Beschreibung |
|----|--------|--------------|
| 0 | ball | Floorball |
| 1 | cornerpoints | Eckpunkte des Spielfelds |
| 2 | goal | Tor |
| 3 | goalkeeper | Torhüter |
| 4 | period | Periodenanzeige |
| 5 | player | Feldspieler |
| 6 | ref | Schiedsrichter |
| 7 | scoreboard | Anzeigetafel |
| 8 | time | Zeitanzeige |

## Projektstruktur

```
floorball_vision/
├── src/                          # Hauptcode
│   ├── data/                     # Datenakquise & Preprocessing
│   │   ├── youtube_downloader.py
│   │   ├── screenshot_extractor.py
│   │   ├── batch_creator.py
│   │   └── data_splitter.py
│   │
│   ├── training/                 # Modell-Training
│   │   └── train.py
│   │
│   ├── trackers/                 # Object Tracking
│   │   ├── player_tracker.py
│   │   └── ball_tracker.py
│   │
│   ├── analysis/                 # Spielanalyse
│   │   ├── team_assigner.py
│   │   ├── ball_possession.py
│   │   └── pass_detector.py
│   │
│   ├── visualization/            # Visualisierung
│   │   ├── tactical_view.py
│   │   └── drawers/
│   │
│   ├── utils/                    # Hilfsfunktionen
│   │   ├── video_utils.py
│   │   └── bbox_utils.py
│   │
│   └── web/                      # Flask Web-App
│       ├── app.py
│       ├── templates/
│       └── static/
│
├── data/                         # Daten (gitignored)
│   ├── raw_videos/               # Heruntergeladene Videos
│   ├── screenshots/              # Extrahierte Frames
│   ├── labeled/                  # Gelabelte Daten
│   └── training/                 # Train/Val/Test Split
│
├── models/                       # Trainierte Modelle (gitignored)
│
├── configs/                      # Konfigurationsdateien
│   ├── classes.yaml
│   └── training.yaml
│
├── notebooks/                    # Jupyter Notebooks (Experimente)
│
├── tests/                        # Unit Tests
│
├── docs/                         # Dokumentation
│
├── requirements.txt
├── setup.py
├── PROJECT.md
└── README.md
```

## Tech Stack

- **Python 3.10+**
- **YOLO** (Ultralytics) - Object Detection
- **OpenCV** - Video/Bild-Verarbeitung
- **ByteTrack** - Multi-Object Tracking
- **Flask** - Web Framework
- **yt-dlp** - YouTube Downloads
- **Label Studio** - Annotation Tool
- **PyTorch** - Deep Learning Backend

## Deployment-Architektur (geplant)

```
┌─────────────────┐     ┌─────────────────┐
│  Lokale Maschine │     │     AWS EC2     │
│  (GPU Training)  │────▶│   (Inference)   │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Flask Web App  │
                        │   (Dashboard)   │
                        └─────────────────┘
```

**Option A**: Vollständig lokal (mit GPU)
**Option B**: Hybrid - Training lokal, Inference auf AWS
**Option C**: Vollständig AWS (mit GPU-Instance für Training)

## Quick Start

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. YouTube-Videos herunterladen
python -m src.data.youtube_downloader --urls data/youtube_links.txt

# 3. Screenshots extrahieren
python -m src.data.screenshot_extractor --input data/raw_videos --output data/screenshots

# 4. Nach dem Labeling: Daten splitten
python -m src.data.data_splitter --input data/labeled --output data/training

# 5. Modell trainieren
python -m src.training.train --config configs/training.yaml

# 6. Video analysieren
python -m src.analyze --video input.mp4 --output output.mp4
```

## Labeling Workflow

1. Screenshots in Label Studio importieren
2. Bounding Boxes für alle 9 Klassen zeichnen
3. Export als YOLO-Format
4. Mit `data_splitter.py` in Train/Val/Test aufteilen

## Referenz

Dieses Projekt orientiert sich an der Architektur des Basketball-Analyse-Projekts, angepasst für Floorball-spezifische Anforderungen.
