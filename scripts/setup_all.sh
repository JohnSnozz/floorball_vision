#!/bin/bash
# ==============================================
# FLOORBALL VISION - Komplettes Setup
# ==============================================
# Führe dieses Script aus, um alles einzurichten:
#   chmod +x scripts/setup_all.sh
#   ./scripts/setup_all.sh
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  FLOORBALL VISION - Setup"
echo "=============================================="
echo ""

# 1. Python Dependencies
echo "=== 1. Python Dependencies installieren ==="
pip install -r requirements.txt
echo "✓ Python Dependencies installiert"
echo ""

# 2. Redis Setup
echo "=== 2. Redis Setup ==="
if command -v redis-cli &> /dev/null && redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo "✓ Redis läuft bereits"
else
    bash "$SCRIPT_DIR/setup_redis.sh"
fi
echo ""

# 3. PostgreSQL Datenbank
echo "=== 3. PostgreSQL Datenbank ==="
echo "Prüfe Datenbankverbindung..."
if PGPASSWORD=postgres psql -h localhost -U postgres -d floorball_vision -c "SELECT 1;" &>/dev/null; then
    echo "✓ Datenbank 'floorball_vision' existiert"
else
    echo "Datenbank muss erstellt werden."
    echo "Führe folgende Befehle manuell aus:"
    echo ""
    echo "  sudo -u postgres psql"
    echo "  CREATE DATABASE floorball_vision;"
    echo "  \\q"
    echo ""
    echo "Oder mit Docker:"
    echo "  cd docker && docker-compose up -d db"
    echo ""
    read -p "Drücke Enter wenn die Datenbank erstellt wurde..."
fi
echo ""

# 4. Alembic Migration
echo "=== 4. Datenbankschema erstellen ==="
echo "Führe Alembic Migration aus..."
cd "$PROJECT_DIR"
if alembic upgrade head; then
    echo "✓ Datenbankschema erstellt"
else
    echo "WARNUNG: Migration fehlgeschlagen. Prüfe die Datenbankverbindung."
fi
echo ""

# 5. Ordnerstruktur prüfen
echo "=== 5. Ordnerstruktur prüfen ==="
mkdir -p data/{videos,frames,labeling/exports,labeling/uploads,training,cache,exports}
mkdir -p models/{base,trained,active}
echo "✓ Ordnerstruktur erstellt"
echo ""

# 6. Abschluss
echo "=============================================="
echo "  Setup abgeschlossen!"
echo "=============================================="
echo ""
echo "Nächste Schritte:"
echo ""
echo "1. Label Studio starten (falls noch nicht läuft):"
echo "   label-studio start --port 8080"
echo ""
echo "2. Label Studio API Key in .env eintragen:"
echo "   - Browser: http://localhost:8080"
echo "   - Settings → Account → Access Token"
echo "   - Token in .env bei LABEL_STUDIO_API_KEY eintragen"
echo ""
echo "3. Flask App starten (Phase 1):"
echo "   flask run"
echo ""
