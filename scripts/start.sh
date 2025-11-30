#!/bin/bash
# ==============================================
# FLOORBALL VISION - Alle Services starten
# ==============================================
# Startet alle benötigten Services:
# - PostgreSQL
# - Redis
# - Label Studio
# - Celery Worker
# - Flask App
#
# Nutzung:
#   chmod +x scripts/start.sh
#   ./scripts/start.sh
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           FLOORBALL VISION - Startup                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# .env laden
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}✓${NC} .env geladen"
else
    echo -e "${RED}✗${NC} .env nicht gefunden! Kopiere .env.example nach .env"
    exit 1
fi

# Cleanup bei Exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping services...${NC}"

    # Flask stoppen (falls im Hintergrund)
    if [ ! -z "$FLASK_PID" ]; then
        kill $FLASK_PID 2>/dev/null || true
    fi

    # Celery stoppen
    if [ ! -z "$CELERY_PID" ]; then
        kill $CELERY_PID 2>/dev/null || true
    fi

    # Label Studio stoppen
    if [ ! -z "$LABEL_STUDIO_PID" ]; then
        kill $LABEL_STUDIO_PID 2>/dev/null || true
    fi

    echo -e "${GREEN}Services gestoppt.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# === 1. PostgreSQL ===
echo -e "\n${YELLOW}[1/5]${NC} PostgreSQL..."
if pg_isready -q 2>/dev/null; then
    echo -e "      ${GREEN}✓${NC} PostgreSQL läuft bereits"
else
    echo -e "      ${YELLOW}→${NC} Versuche PostgreSQL zu starten..."
    sudo service postgresql start 2>/dev/null || true
    sleep 2
    if pg_isready -q 2>/dev/null; then
        echo -e "      ${GREEN}✓${NC} PostgreSQL gestartet"
    else
        echo -e "      ${RED}✗${NC} PostgreSQL nicht erreichbar"
        echo -e "      ${YELLOW}Tipp:${NC} Starte manuell: sudo service postgresql start"
        exit 1
    fi
fi

# === 2. Redis ===
echo -e "\n${YELLOW}[2/5]${NC} Redis..."
if redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo -e "      ${GREEN}✓${NC} Redis läuft bereits"
else
    sudo service redis-server start
    sleep 1
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo -e "      ${GREEN}✓${NC} Redis gestartet"
    else
        echo -e "      ${RED}✗${NC} Redis konnte nicht gestartet werden"
        exit 1
    fi
fi

# === 3. Label Studio ===
echo -e "\n${YELLOW}[3/5]${NC} Label Studio..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "      ${GREEN}✓${NC} Label Studio läuft bereits auf Port 8080"
else
    echo -e "      ${YELLOW}→${NC} Starte Label Studio im Hintergrund..."

    # Label Studio im Hintergrund starten
    nohup label-studio start --port 8080 > /tmp/label-studio.log 2>&1 &
    LABEL_STUDIO_PID=$!

    # Warten bis Label Studio bereit ist
    echo -n "      Warte auf Label Studio"
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo ""
            echo -e "      ${GREEN}✓${NC} Label Studio gestartet (PID: $LABEL_STUDIO_PID)"
            break
        fi
        echo -n "."
        sleep 2
    done

    if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo ""
        echo -e "      ${RED}✗${NC} Label Studio konnte nicht gestartet werden"
        echo "      Log: /tmp/label-studio.log"
    fi
fi

# === 4. Celery Worker ===
echo -e "\n${YELLOW}[4/5]${NC} Celery Worker..."

# Alten Celery Worker beenden falls vorhanden
pkill -f "celery.*worker" 2>/dev/null || true
sleep 1

echo -e "      ${YELLOW}→${NC} Starte Celery Worker im Hintergrund..."
cd "$PROJECT_DIR"
nohup celery -A src.web.extensions:celery worker --loglevel=info > /tmp/celery.log 2>&1 &
CELERY_PID=$!
sleep 2

if ps -p $CELERY_PID > /dev/null 2>&1; then
    echo -e "      ${GREEN}✓${NC} Celery Worker gestartet (PID: $CELERY_PID)"
    echo -e "      Log: /tmp/celery.log"
else
    echo -e "      ${RED}✗${NC} Celery Worker konnte nicht gestartet werden"
    echo "      Prüfe Log: /tmp/celery.log"
fi

# === 5. Flask App ===
echo -e "\n${YELLOW}[5/5]${NC} Flask App..."
echo -e "      ${YELLOW}→${NC} Starte Flask Development Server..."

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    SERVICES GESTARTET                      ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}Flask App:${NC}      http://localhost:5000               ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}Label Studio:${NC}  http://localhost:8080               ${BLUE}║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║${NC}  ${YELLOW}Logs:${NC}                                               ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}    - Label Studio: /tmp/label-studio.log              ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}    - Celery:       /tmp/celery.log                    ${BLUE}║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║${NC}  ${RED}Ctrl+C zum Beenden aller Services${NC}                   ${BLUE}║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Flask im Vordergrund starten (damit Ctrl+C funktioniert)
cd "$PROJECT_DIR"
python -m src.web.app
