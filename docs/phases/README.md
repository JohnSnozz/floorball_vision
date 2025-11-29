# Phasen-Übersicht

Dieses Verzeichnis enthält die detaillierten Schritt-für-Schritt-Anleitungen für jede Phase.

## Status

| Phase | Datei | Status |
|-------|-------|--------|
| 0 | [PHASE_0_SETUP.md](PHASE_0_SETUP.md) | **IN ARBEIT** |
| 1 | [PHASE_1_WEB.md](PHASE_1_WEB.md) | Ausstehend |
| 1.5 | [PHASE_1_5_LABELSTUDIO.md](PHASE_1_5_LABELSTUDIO.md) | Ausstehend |
| 2 | [PHASE_2_CALIBRATION.md](PHASE_2_CALIBRATION.md) | Ausstehend |
| 3 | [PHASE_3_TRACKING.md](PHASE_3_TRACKING.md) | Ausstehend |
| 4 | [PHASE_4_PREVIEW.md](PHASE_4_PREVIEW.md) | Ausstehend |
| 5 | [PHASE_5_FULLANALYSIS.md](PHASE_5_FULLANALYSIS.md) | Ausstehend |
| 6 | [PHASE_6_EXPORT.md](PHASE_6_EXPORT.md) | Ausstehend |

## Wie die Phasen funktionieren

1. **Jede Phase hat klare Grenzen** - Nur bestimmte Dateien dürfen geändert werden
2. **Tests müssen bestehen** - Bevor zur nächsten Phase gewechselt wird
3. **CLAUDE.md ist die Quelle der Wahrheit** - Dort steht die aktuelle Phase
4. **Git Tags markieren Phasen-Abschlüsse** - `phase-X-complete`

## Schnellstart

```bash
# Aktuelle Phase anzeigen
grep "AKTUELLE PHASE" ../CLAUDE.md

# Zur Phasen-Dokumentation springen
cat docs/phases/PHASE_0_SETUP.md
```
