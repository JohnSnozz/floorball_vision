#!/bin/bash
# ==============================================
# FLOORBALL VISION - Datenbank Setup
# ==============================================
# Führe dieses Script aus, um die Datenbank zu erstellen:
#   chmod +x scripts/setup_db.sh
#   ./scripts/setup_db.sh
# ==============================================

set -e

echo "=== Floorball Vision - Database Setup ==="
echo ""

# Prüfe ob PostgreSQL läuft
if ! pg_isready -q; then
    echo "ERROR: PostgreSQL läuft nicht!"
    echo "Starte PostgreSQL mit: sudo service postgresql start"
    exit 1
fi

echo "✓ PostgreSQL läuft"

# Datenbank erstellen (falls nicht vorhanden)
echo ""
echo "Erstelle Datenbank 'floorball_vision'..."
echo "(Falls du nach einem Passwort gefragt wirst, gib das PostgreSQL-Passwort ein)"
echo ""

# Versuche die Datenbank zu erstellen
sudo -u postgres psql -c "CREATE DATABASE floorball_vision;" 2>/dev/null || echo "Datenbank existiert bereits"

# User erstellen (optional, falls du einen separaten User willst)
# sudo -u postgres psql -c "CREATE USER floorball WITH PASSWORD 'floorball123';"
# sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE floorball_vision TO floorball;"

echo ""
echo "✓ Datenbank Setup abgeschlossen"
echo ""
echo "Nächster Schritt: Alembic Migration ausführen"
echo "  cd /home/jonas/floorball_vision"
echo "  alembic upgrade head"
