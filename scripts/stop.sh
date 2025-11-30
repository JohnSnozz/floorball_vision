#!/bin/bash
# ==============================================
# FLOORBALL VISION - Alle Services stoppen
# ==============================================

echo "Stopping Floorball Vision services..."

# Celery Worker stoppen
echo "Stopping Celery..."
pkill -f "celery.*worker" 2>/dev/null || true

# Label Studio stoppen (optional)
read -p "Label Studio auch stoppen? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pkill -f "label-studio" 2>/dev/null || true
    echo "Label Studio gestoppt"
fi

echo ""
echo "Services gestoppt."
echo ""
echo "PostgreSQL und Redis laufen weiter (Systemdienste)."
echo "Falls gewünscht:"
echo "  sudo service postgresql stop"
echo "  sudo service redis-server stop"
