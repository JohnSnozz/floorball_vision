# Phase 0: Setup & Infrastruktur

## Ziel
Entwicklungsumgebung mit PostgreSQL, Redis und Label Studio vollständig einrichten.

## Voraussetzungen
- [x] PostgreSQL 16 installiert
- [x] Label Studio installiert (Anaconda)
- [ ] Redis installiert
- [ ] Docker installiert

---

## Schritt-für-Schritt Plan

### Schritt 0.1: Docker Compose erstellen
**Datei:** `docker/docker-compose.yml`

```bash
# Erstellen
mkdir -p docker
# docker-compose.yml wird erstellt
```

**Test:**
```bash
cd docker
docker-compose config  # Keine Fehler
```

---

### Schritt 0.2: PostgreSQL Datenbank erstellen

```bash
# Option A: Mit Docker
docker-compose up -d db
docker-compose exec db psql -U postgres -c "CREATE DATABASE floorball_vision;"

# Option B: Lokal (falls PostgreSQL bereits läuft)
sudo -u postgres psql -c "CREATE DATABASE floorball_vision;"
sudo -u postgres psql -c "CREATE USER floorball WITH PASSWORD 'floorball123';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE floorball_vision TO floorball;"
```

**Test:**
```bash
psql -h localhost -U postgres -d floorball_vision -c "SELECT 1 AS test;"
# Erwartet: test = 1
```

---

### Schritt 0.3: Redis installieren/starten

```bash
# Option A: Mit Docker (empfohlen)
docker-compose up -d redis

# Option B: Lokal installieren
sudo apt install redis-server
sudo systemctl start redis
```

**Test:**
```bash
redis-cli ping
# Erwartet: PONG
```

---

### Schritt 0.4: .env Datei erstellen
**Datei:** `.env`

```bash
cp .env.example .env
# Werte anpassen
```

**Test:**
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('DATABASE_URL'))"
# Erwartet: postgresql://...
```

---

### Schritt 0.5: Requirements aktualisieren
**Datei:** `requirements.txt`

Neue Abhängigkeiten:
- `flask`
- `flask-sqlalchemy`
- `psycopg2-binary`
- `celery[redis]`
- `python-dotenv`
- `alembic`

**Test:**
```bash
pip install -r requirements.txt
python -c "import flask; import celery; import sqlalchemy; print('OK')"
```

---

### Schritt 0.6: Alembic Migrations Setup
**Ordner:** `migrations/`

```bash
# Initialisieren
alembic init migrations

# alembic.ini und migrations/env.py anpassen
```

**Test:**
```bash
alembic current
# Erwartet: (keine Migration, aber kein Fehler)
```

---

### Schritt 0.7: Erste Migration erstellen
**Datei:** `migrations/versions/001_initial.py`

Tabellen:
- videos
- calibrations
- labeling_projects
- training_runs
- active_model
- analysis_jobs

**Test:**
```bash
alembic upgrade head
psql -h localhost -U postgres -d floorball_vision -c "\dt"
# Erwartet: Alle Tabellen werden aufgelistet
```

---

### Schritt 0.8: Label Studio API Key generieren

```bash
# Label Studio starten
label-studio start --port 8080

# Im Browser:
# 1. http://localhost:8080 öffnen
# 2. Account erstellen/einloggen
# 3. Settings → Account & Settings → Access Token
# 4. Token kopieren und in .env einfügen
```

**Test:**
```bash
curl -H "Authorization: Token $LABEL_STUDIO_API_KEY" \
     http://localhost:8080/api/projects
# Erwartet: [] (leere Liste, kein Fehler)
```

---

### Schritt 0.9: Configs erstellen

**Dateien:**
- `configs/classes.yaml`
- `configs/field_dimensions.yaml`
- `configs/label_studio.yaml`
- `configs/training_defaults.yaml`

**Test:**
```bash
python -c "import yaml; yaml.safe_load(open('configs/classes.yaml')); print('OK')"
```

---

### Schritt 0.10: Data-Ordner Struktur

```bash
mkdir -p data/{videos,frames,labeling/exports,labeling/uploads,training,cache,exports}
mkdir -p models/{base,trained,active}
```

**Test:**
```bash
ls -la data/
# Erwartet: Alle Unterordner vorhanden
```

---

## Abschluss-Checkliste

- [ ] PostgreSQL läuft und Datenbank existiert
- [ ] Redis läuft
- [ ] Label Studio läuft auf Port 8080
- [ ] `.env` ist konfiguriert
- [ ] Alle Python-Abhängigkeiten installiert
- [ ] Alembic Migrations erfolgreich
- [ ] Alle Tabellen in DB erstellt
- [ ] Config-Dateien vorhanden
- [ ] Data-Ordner Struktur erstellt

## Finale Tests

```bash
# Alle Services prüfen
docker-compose ps                           # db, redis: Up
label-studio --version                      # Version angezeigt
psql -d floorball_vision -c "\dt"          # Tabellen aufgelistet
redis-cli ping                              # PONG
python -c "from dotenv import load_dotenv; load_dotenv(); import os; assert os.getenv('DATABASE_URL')"

echo "Phase 0 abgeschlossen!"
```

---

## Nach Abschluss

1. CLAUDE.md aktualisieren: "AKTUELLE PHASE: 1"
2. Git Commit: `git commit -m "Phase 0: Setup & Infrastruktur abgeschlossen"`
3. Git Tag: `git tag phase-0-complete`
