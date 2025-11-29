# Phase 5: Vollständige Analyse

## Ziel
Ganzes Video analysieren und Daten in Datenbank speichern.

## Voraussetzungen
- [x] Phase 4 abgeschlossen
- [x] Preview funktioniert zufriedenstellend

---

## Schritt-für-Schritt Plan

### Schritt 5.1: Full Analysis Template
**Datei:** `src/web/templates/analysis/full.html`

- Konfiguration (Sample Rate, Zeitbereich)
- Progress-Anzeige
- Live Preview
- Statistiken

### Schritt 5.2: Full Analysis Celery Task
**Datei:** `src/processing/tasks.py` (erweitern)

- `full_analysis(job_id)`
- Batch Processing
- Progress Updates
- Checkpoint/Resume

### Schritt 5.3: DB Utils für Bulk Insert
**Datei:** `src/utils/db_utils.py`

- `bulk_insert_positions(positions)`
- Effizientes Batch-Schreiben
- Partitionierung

### Schritt 5.4: Full Analysis API Endpoints
**Datei:** `src/web/routes/analysis.py` (erweitern)

- `POST /api/analysis/full` - Analyse starten
- `GET /api/analysis/jobs/<id>` - Status
- `POST /api/analysis/jobs/<id>/pause` - Pausieren
- `POST /api/analysis/jobs/<id>/resume` - Fortsetzen

---

## Tests

```bash
# 1. Full Analysis starten
curl -X POST http://localhost:5000/api/analysis/full \
     -d '{"video_id": "...", "calibration_id": "...", "sample_rate": 5}'
# Job wird erstellt

# 2. Status prüfen
curl http://localhost:5000/api/analysis/jobs/{id}
# Progress wird angezeigt

# 3. Daten in DB prüfen
psql -c "SELECT COUNT(*) FROM player_positions WHERE job_id = '...'"
# Daten sind vorhanden

# 4. Analyse abschliessen
# Status = "done", completed_at gesetzt
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 6"
2. Git Commit: `git commit -m "Phase 5: Vollständige Analyse abgeschlossen"`
3. Git Tag: `git tag phase-5-complete`
