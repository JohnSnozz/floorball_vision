# Phase 6: Export & Visualisierung

## Ziel
Analysierte Daten exportieren und visualisieren.

## Voraussetzungen
- [x] Phase 5 abgeschlossen
- [x] Analysierte Daten vorhanden

---

## Schritt-für-Schritt Plan

### Schritt 6.1: Export Module
**Dateien:**
- `src/export/__init__.py`
- `src/export/csv_export.py`
- `src/export/parquet_export.py`
- `src/export/json_export.py`

### Schritt 6.2: Export Routes
**Datei:** `src/web/routes/export.py`

- `GET /api/export/jobs/<id>/csv`
- `GET /api/export/jobs/<id>/parquet`
- `GET /api/export/jobs/<id>/json`

### Schritt 6.3: Export Template
**Datei:** `src/web/templates/export/download.html`

- Format-Auswahl
- Zeitbereich-Filter
- Download-Links

### Schritt 6.4: Tactical View JavaScript
**Datei:** `src/web/static/js/tactical-view.js`

- Spielfeld-Zeichnung
- Spieler-Positionen animieren
- Play/Pause/Seek

### Schritt 6.5: Visualization Template
**Datei:** `src/web/templates/visualization/tactical.html`

- Tactical View Canvas
- Steuerung
- Spieler-Auswahl

---

## Tests

```bash
# 1. CSV Export
curl http://localhost:5000/api/export/jobs/{id}/csv -o export.csv
# CSV Datei wird heruntergeladen

# 2. CSV validieren
head export.csv
# Korrekte Spalten und Daten

# 3. Tactical View
# Browser: /visualization/{job_id}
# Spieler werden animiert angezeigt
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: ABGESCHLOSSEN"
2. Git Commit: `git commit -m "Phase 6: Export & Visualisierung abgeschlossen"`
3. Git Tag: `git tag phase-6-complete`
4. Git Tag: `git tag v1.0.0`

## PROJEKT ABGESCHLOSSEN!
