# Session Status - Floorball Vision

**Letzte Aktualisierung:** 29. November 2025

## Aktuelle Phase
**Phase 1.5: Label Studio Integration** - IN ARBEIT

## Abgeschlossene Aufgaben

1. **Projekt unabhängig von Video machen** - Ein Projekt kann jetzt mehrere Batches aus verschiedenen Videos haben
2. **UI anpassen für mehrere Batches** - Projekt-Detailseite zeigt Batches, neue Buttons zum Erstellen
3. **API-Key in .env eintragen** - JWT Token konfiguriert
4. **JWT Token Auto-Refresh implementieren** - Label Studio 1.20 verwendet JWT, Client refresht automatisch
5. **Batch erstellen & öffnen Feature** - Button "Batch erstellen & in Label Studio öffnen" erstellt einen gefilterten Tab/View

## Offene Aufgaben

### NÄCHSTER SCHRITT: Labeling-Fortschritt korrigieren
- **Problem:** Auf der Projekt-Detailseite (`/labeling/project/<id>`) zeigt "Labeling-Fortschritt" nicht die korrekten Zahlen an
- Der User hat bereits 150 Bilder gelabelt, aber das wird nicht angezeigt
- **TODO:** Die `list_tasks` Funktion prüft ob Tasks Annotations haben, aber die Anzeige stimmt nicht
- **Datei:** `src/web/routes/labeling.py` - Funktion `project_detail()` (Zeile 38-69) holt `ls_info`
- **Template:** `src/web/templates/labeling/project.html` - Zeigt `ls_info.total_tasks`, `ls_info.completed_tasks`, `ls_info.progress`

### Weitere offene Punkte Phase 1.5
- [ ] Labels exportieren testen (YOLO-Format)
- [ ] Training starten testen
- [ ] Training-Templates fertigstellen (`src/web/templates/training/`)

## Technische Details

### Label Studio Integration
- **URL:** http://localhost:8080
- **Auth:** JWT Token (Personal Access Token) in `.env` als `LABEL_STUDIO_API_KEY`
- **Client:** `src/labeling/client.py` - Konvertiert automatisch PAT zu Access Token

### Batch-System
- Batches werden mit `LabelingSync.extract_random_frames()` erstellt
- Beim Upload werden Task-IDs gesammelt
- Ein Label Studio **View/Tab** wird erstellt mit Filter für die Task-IDs
- View-ID wird in `labeling_batches.view_id` gespeichert
- URL zum Öffnen: `http://localhost:8080/projects/{project_id}/data?tab={view_id}`

### Datenbank-Änderungen (Migrationen angewendet)
- `labeling_batches.task_ids` - JSON Array der Task-IDs
- `labeling_batches.view_id` - Label Studio View-ID

## Dateien die geändert wurden

- `src/labeling/client.py` - JWT Auth, View-Erstellung
- `src/labeling/sync.py` - Task-ID Tracking beim Upload
- `src/web/routes/labeling.py` - Neue Endpoints für Batch-Erstellung
- `src/web/templates/labeling/project.html` - UI für Batches
- `src/web/templates/labeling/index.html` - UI für Projekt-Verknüpfung
- `src/db/models.py` - task_ids und view_id Spalten

## Befehle zum Starten

```bash
# Flask App
python -m src.web.app

# Label Studio (falls nicht läuft)
label-studio start --port 8080

# Celery Worker (für Training später)
celery -A src.processing.tasks worker --loglevel=info
```
