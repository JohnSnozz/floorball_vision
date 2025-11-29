# Phase 4: Snippet-Preview

## Ziel
Kurze Video-Ausschnitte analysieren zur Qualitätskontrolle.

## Voraussetzungen
- [x] Phase 3 abgeschlossen
- [x] Tracking funktioniert

---

## Schritt-für-Schritt Plan

### Schritt 4.1: Video Player JavaScript
**Datei:** `src/web/static/js/video-player.js`

- HTML5 Video Steuerung
- Timeline mit Snippet-Markierungen
- Frame-genaues Seeking

### Schritt 4.2: Snippet Selection JavaScript
**Datei:** `src/web/static/js/snippets.js`

- In/Out Punkte setzen
- Mehrere Snippets verwalten
- Visuelles Feedback

### Schritt 4.3: Preview Template
**Datei:** `src/web/templates/analysis/preview.html`

- Video Player
- Snippet-Auswahl
- Ergebnis-Anzeige

### Schritt 4.4: Preview Celery Task
**Datei:** `src/processing/tasks.py` (erweitern)

- `preview_analysis(video_id, start_time, end_time)`

### Schritt 4.5: Preview API Endpoints
**Datei:** `src/web/routes/analysis.py` (erweitern)

- `POST /api/analysis/preview` - Preview starten
- `GET /api/analysis/preview/<id>` - Status/Ergebnis

---

## Tests

```bash
# 1. Video Player lädt
# Browser: /videos/{id}/preview

# 2. Snippet auswählen
# In/Out Punkte setzen, Snippet wird gespeichert

# 3. Preview Analyse
curl -X POST http://localhost:5000/api/analysis/preview \
     -d '{"video_id": "...", "start_time": 30, "end_time": 40}'
# Job wird erstellt

# 4. Ergebnisse abrufen
curl http://localhost:5000/api/analysis/preview/{id}
# Annotierte Frames und Statistiken
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 5"
2. Git Commit: `git commit -m "Phase 4: Snippet-Preview abgeschlossen"`
3. Git Tag: `git tag phase-4-complete`
