# Phase 1.5: Label Studio Integration

## Ziel
Label Studio mit der App verbinden, Screenshots hochladen, Training automatisieren.

## Voraussetzungen
- [x] Phase 1 abgeschlossen
- [x] Label Studio läuft auf Port 8080
- [x] API Key in .env

---

## Schritt-für-Schritt Plan

### Schritt 1.5.1: Label Studio Client
**Datei:** `src/labeling/client.py`

Klasse `LabelStudioClient`:
- `create_project(name, label_config)`
- `get_projects()`
- `upload_images(project_id, images)`
- `get_annotations(project_id)`
- `export_yolo(project_id, output_dir)`
- `get_project_stats(project_id)`

**Test:**
```bash
python -c "
from src.labeling.client import LabelStudioClient
client = LabelStudioClient()
projects = client.get_projects()
print(f'Found {len(projects)} projects')
"
```

---

### Schritt 1.5.2: YOLO Export Handler
**Datei:** `src/labeling/export.py`

Funktionen:
- `export_to_yolo(project_id, output_dir)`
- `convert_annotations(annotations)`
- `create_data_yaml(output_dir, classes)`

**Test:**
```bash
# Nach manuellem Labeling in Label Studio:
python -c "
from src.labeling.export import export_to_yolo
export_to_yolo(1, 'data/labeling/exports/test')
"
ls data/labeling/exports/test/
# Erwartet: images/, labels/, data.yaml
```

---

### Schritt 1.5.3: Labeling Routes
**Datei:** `src/web/routes/labeling.py`

Endpoints:
- `GET /api/labeling/projects` - Projekte auflisten
- `POST /api/labeling/projects` - Neues Projekt
- `GET /api/labeling/projects/<id>/stats` - Fortschritt
- `POST /api/labeling/projects/<id>/upload` - Screenshots hochladen
- `POST /api/labeling/projects/<id>/export` - YOLO Export

**Test:**
```bash
curl http://localhost:5000/api/labeling/projects
# Erwartet: Liste der Projekte
```

---

### Schritt 1.5.4: Training Routes
**Datei:** `src/web/routes/training.py`

Endpoints:
- `GET /api/training/runs` - Training-Historie
- `POST /api/training/runs` - Training starten
- `GET /api/training/runs/<id>` - Training Status
- `POST /api/training/active` - Aktives Modell setzen

**Test:**
```bash
curl http://localhost:5000/api/training/runs
# Erwartet: []
```

---

### Schritt 1.5.5: YOLO Trainer
**Datei:** `src/training/trainer.py`

Klasse `YOLOTrainer`:
- `prepare_dataset(export_dir, output_dir)`
- `train(data_yaml, epochs, batch_size)`
- `get_metrics(run_dir)`

**Test:**
```bash
python -c "
from src.training.trainer import YOLOTrainer
trainer = YOLOTrainer('yolov8n.pt')
print('Trainer initialized')
"
```

---

### Schritt 1.5.6: Training Celery Task
**Datei:** `src/processing/tasks.py` (erweitern)

Task:
- `start_training(training_run_id)`

**Test:**
```bash
# Im Celery Worker Log sollte der Task registriert sein
```

---

### Schritt 1.5.7: Labeling Templates
**Dateien:**
- `src/web/templates/labeling/projects.html`
- `src/web/templates/training/dashboard.html`

**Test:**
```bash
# Browser: http://localhost:5000/labeling
# - Projekte werden angezeigt
# - "In Label Studio öffnen" Link funktioniert
```

---

### Schritt 1.5.8: Screenshot Upload zu Label Studio
**Datei:** `src/labeling/sync.py`

Funktionen:
- `upload_screenshots(video_id, project_id, count)`
- `sync_project_stats(project_id)`

**Test:**
```bash
# In UI: Video auswählen → Screenshots generieren → Hochladen
# In Label Studio: Bilder sind im Projekt sichtbar
```

---

## Finale Tests

```bash
# 1. Neues Projekt erstellen
curl -X POST http://localhost:5000/api/labeling/projects \
     -H "Content-Type: application/json" \
     -d '{"name": "test_project"}'

# 2. Screenshots hochladen
curl -X POST http://localhost:5000/api/labeling/projects/1/upload \
     -H "Content-Type: application/json" \
     -d '{"video_id": "...", "count": 10}'

# 3. In Label Studio labeln...

# 4. Export
curl -X POST http://localhost:5000/api/labeling/projects/1/export

# 5. Training starten
curl -X POST http://localhost:5000/api/training/runs \
     -H "Content-Type: application/json" \
     -d '{"project_id": 1, "epochs": 10}'

# 6. Training Status
curl http://localhost:5000/api/training/runs/1
# Erwartet: {"status": "running", "progress": 0.5, ...}

echo "Phase 1.5 abgeschlossen!"
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 2"
2. Git Commit: `git commit -m "Phase 1.5: Label Studio Integration abgeschlossen"`
3. Git Tag: `git tag phase-1.5-complete`
