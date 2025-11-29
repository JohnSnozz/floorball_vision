#!/bin/bash
# ==============================================
# FLOORBALL VISION - Redis Setup
# ==============================================
# Führe dieses Script aus, um Redis zu installieren:
#   chmod +x scripts/setup_redis.sh
#   ./scripts/setup_redis.sh
# ==============================================

set -e

echo "=== Floorball Vision - Redis Setup ==="
echo ""

# Prüfe ob Redis bereits installiert ist
if command -v redis-cli &> /dev/null; then
    echo "✓ Redis ist bereits installiert"
    redis-cli --version
else
    echo "Redis wird installiert..."
    sudo apt update
    sudo apt install -y redis-server
    echo "✓ Redis installiert"
fi

# Redis starten
echo ""
echo "Starte Redis..."
sudo service redis-server start || sudo systemctl start redis-server

# Prüfe ob Redis läuft
sleep 1
if redis-cli ping | grep -q "PONG"; then
    echo "✓ Redis läuft"
else
    echo "ERROR: Redis konnte nicht gestartet werden"
    exit 1
fi

echo ""
echo "✓ Redis Setup abgeschlossen"
