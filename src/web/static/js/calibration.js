/**
 * Calibration Canvas - Interaktives Zeichnen von Punkten und Polygonen
 *
 * Features:
 * - Banden-Polygon zeichnen
 * - Referenzpunkte auf Screenshot setzen
 * - Referenzpunkte auf Spielfeld-Template setzen
 * - Undo/Redo
 * - Zoom/Pan
 */

class CalibrationCanvas {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        this.options = {
            mode: options.mode || 'points', // 'points' oder 'polygon'
            maxPoints: options.maxPoints || null,
            pointColor: options.pointColor || '#FF6B00',
            polygonColor: options.polygonColor || '#00AAFF',
            pointRadius: options.pointRadius || 8,
            lineWidth: options.lineWidth || 2,
            onPointAdded: options.onPointAdded || null,
            onPointRemoved: options.onPointRemoved || null,
            onPolygonComplete: options.onPolygonComplete || null,
        };

        this.points = [];
        this.polygonPoints = [];
        this.straightLines = [];  // Array von Linien, jede Linie = Array von Punkten
        this.currentStraightLine = [];  // Aktuelle Linie die gerade gezeichnet wird
        this.history = [];
        this.historyIndex = -1;

        this.image = null;
        this.imageLoaded = false;

        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;

        this.isDragging = false;
        this.dragPointIndex = -1;
        this.dragPolygonIndex = -1;

        this.hoveredPointIndex = -1;

        // Pan-Modus (Leertaste gedrückt)
        this.isPanMode = false;
        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;

        this._setupEventListeners();
    }

    /**
     * Lädt ein Bild in den Canvas
     */
    loadImage(src) {
        return new Promise((resolve, reject) => {
            this.image = new Image();
            this.image.onload = () => {
                this.imageLoaded = true;
                this._fitImageToCanvas();
                this.render();
                resolve();
            };
            this.image.onerror = reject;
            this.image.src = src;
        });
    }

    /**
     * Setzt den Modus (points oder polygon)
     */
    setMode(mode) {
        this.options.mode = mode;
        this.render();
    }

    /**
     * Setzt die Punkte
     */
    setPoints(points) {
        this.points = points.map(p => ({ x: p[0], y: p[1] }));
        this._saveHistory();
        this.render();
    }

    /**
     * Gibt die Punkte zurück
     */
    getPoints() {
        return this.points.map(p => [p.x, p.y]);
    }

    /**
     * Setzt das Polygon
     */
    setPolygon(points) {
        this.polygonPoints = points.map(p => ({ x: p[0], y: p[1] }));
        this._saveHistory();
        this.render();
    }

    /**
     * Gibt das Polygon zurück
     */
    getPolygon() {
        return this.polygonPoints.map(p => [p.x, p.y]);
    }

    /**
     * Löscht alle Punkte
     */
    clearPoints() {
        this.points = [];
        this._saveHistory();
        this.render();
    }

    /**
     * Löscht das Polygon
     */
    clearPolygon() {
        this.polygonPoints = [];
        this._saveHistory();
        this.render();
    }

    /**
     * Löscht alles
     */
    clearAll() {
        this.points = [];
        this.polygonPoints = [];
        this._saveHistory();
        this.render();
    }

    /**
     * Undo
     */
    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this._restoreFromHistory();
            this.render();
        }
    }

    /**
     * Redo
     */
    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this._restoreFromHistory();
            this.render();
        }
    }

    /**
     * Letzten Punkt entfernen
     */
    removeLastPoint() {
        if (this.options.mode === 'polygon' && this.polygonPoints.length > 0) {
            this.polygonPoints.pop();
            this._saveHistory();
            this.render();
            if (this.options.onPointRemoved) {
                this.options.onPointRemoved(this.polygonPoints.length);
            }
        } else if (this.options.mode === 'points' && this.points.length > 0) {
            this.points.pop();
            this._saveHistory();
            this.render();
            if (this.options.onPointRemoved) {
                this.options.onPointRemoved(this.points.length);
            }
        }
    }

    /**
     * Punkt an einem bestimmten Index entfernen
     */
    removePointAtIndex(index) {
        if (this.options.mode === 'points' && index >= 0 && index < this.points.length) {
            this.points.splice(index, 1);
            this._saveHistory();
            this.render();
            if (this.options.onPointRemoved) {
                this.options.onPointRemoved(this.points.length);
            }
        }
    }

    /**
     * Rendert den Canvas
     */
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Bild zeichnen
        if (this.imageLoaded && this.image) {
            this.ctx.save();
            this.ctx.translate(this.offsetX, this.offsetY);
            this.ctx.scale(this.scale, this.scale);
            this.ctx.drawImage(this.image, 0, 0);
            this.ctx.restore();
        }

        // Polygon zeichnen
        if (this.polygonPoints.length > 0) {
            this._drawPolygon();
        }

        // Gerade-Linien zeichnen (Hilfslinien für Entzerrung)
        this._drawStraightLines();

        // Punkte zeichnen
        if (this.points.length > 0) {
            this._drawPoints();
        }
    }

    // === Private Methoden ===

    _setupEventListeners() {
        this.canvas.addEventListener('click', (e) => this._handleClick(e));
        this.canvas.addEventListener('mousemove', (e) => this._handleMouseMove(e));
        this.canvas.addEventListener('mousedown', (e) => this._handleMouseDown(e));
        this.canvas.addEventListener('mouseup', (e) => this._handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this._handleMouseLeave(e));
        this.canvas.addEventListener('wheel', (e) => this._handleWheel(e));
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.removeLastPoint();
        });

        // Keyboard Events für Pan-Modus (Leertaste)
        document.addEventListener('keydown', (e) => this._handleKeyDown(e));
        document.addEventListener('keyup', (e) => this._handleKeyUp(e));

        // Touch Events für Mobile
        this.canvas.addEventListener('touchstart', (e) => this._handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this._handleTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this._handleTouchEnd(e));
    }

    _handleKeyDown(e) {
        // Leertaste aktiviert Pan-Modus
        if (e.code === 'Space' && !this.isPanMode) {
            e.preventDefault();
            this.isPanMode = true;
            this.canvas.style.cursor = 'grab';
        }
    }

    _handleKeyUp(e) {
        if (e.code === 'Space') {
            e.preventDefault();
            this.isPanMode = false;
            this.isPanning = false;
            this.canvas.style.cursor = this.hoveredPointIndex >= 0 ? 'grab' : 'crosshair';
        }
    }

    _handleMouseLeave(e) {
        // Panning stoppen wenn Maus Canvas verlässt
        if (this.isPanning) {
            this.isPanning = false;
            this.canvas.style.cursor = this.isPanMode ? 'grab' : 'crosshair';
        }
    }

    _handleClick(e) {
        // Kein Klick im Pan-Modus oder beim Dragging
        if (this.isDragging || this.isPanMode || this.isPanning) return;

        const pos = this._getCanvasPosition(e);
        const imagePos = this._canvasToImage(pos);

        if (this.options.mode === 'polygon') {
            this.polygonPoints.push(imagePos);
            this._saveHistory();
            this.render();

            if (this.options.onPointAdded) {
                this.options.onPointAdded(this.polygonPoints.length, imagePos);
            }
        } else if (this.options.mode === 'straightLine') {
            // Hilfslinien-Modus für Entzerrung
            this.currentStraightLine.push(imagePos);
            this.render();

            if (this.options.onStraightLinePointAdded) {
                this.options.onStraightLinePointAdded(this.currentStraightLine.length, imagePos);
            }
        } else {
            // Punkte-Modus
            if (this.options.maxPoints && this.points.length >= this.options.maxPoints) {
                return;
            }

            this.points.push(imagePos);
            this._saveHistory();
            this.render();

            if (this.options.onPointAdded) {
                this.options.onPointAdded(this.points.length, imagePos);
            }
        }
    }

    _handleMouseMove(e) {
        const pos = this._getCanvasPosition(e);

        // Panning (Leertaste + Maus ziehen)
        if (this.isPanning) {
            const dx = pos.x - this.panStartX;
            const dy = pos.y - this.panStartY;
            this.offsetX += dx;
            this.offsetY += dy;
            this.panStartX = pos.x;
            this.panStartY = pos.y;
            this.render();
            return;
        }

        const imagePos = this._canvasToImage(pos);

        // Im Pan-Modus keine Punkt-Interaktion
        if (this.isPanMode) {
            this.canvas.style.cursor = 'grab';
            return;
        }

        // Hover-Effekt für Punkte
        this.hoveredPointIndex = -1;
        const points = this.options.mode === 'polygon' ? this.polygonPoints : this.points;

        for (let i = 0; i < points.length; i++) {
            const pt = this._imageToCanvas(points[i]);
            const dist = Math.sqrt(Math.pow(pos.x - pt.x, 2) + Math.pow(pos.y - pt.y, 2));
            if (dist < this.options.pointRadius + 5) {
                this.hoveredPointIndex = i;
                break;
            }
        }

        // Cursor ändern
        this.canvas.style.cursor = this.hoveredPointIndex >= 0 ? 'grab' : 'crosshair';

        // Dragging (Punkte verschieben)
        if (this.isDragging && this.dragPointIndex >= 0) {
            if (this.options.mode === 'polygon') {
                this.polygonPoints[this.dragPointIndex] = imagePos;
            } else {
                this.points[this.dragPointIndex] = imagePos;
            }
            this.render();
        }
    }

    _handleMouseDown(e) {
        const pos = this._getCanvasPosition(e);

        // Pan-Modus: Panning starten
        if (this.isPanMode) {
            this.isPanning = true;
            this.panStartX = pos.x;
            this.panStartY = pos.y;
            this.canvas.style.cursor = 'grabbing';
            e.preventDefault();
            return;
        }

        // Punkt verschieben
        if (this.hoveredPointIndex >= 0) {
            this.isDragging = true;
            this.dragPointIndex = this.hoveredPointIndex;
            this.canvas.style.cursor = 'grabbing';
            e.preventDefault();
        }
    }

    _handleMouseUp(e) {
        // Panning beenden
        if (this.isPanning) {
            this.isPanning = false;
            this.canvas.style.cursor = this.isPanMode ? 'grab' : 'crosshair';
            return;
        }

        // Punkt-Dragging beenden
        if (this.isDragging) {
            this.isDragging = false;
            this.dragPointIndex = -1;
            this._saveHistory();
            this.canvas.style.cursor = 'crosshair';
        }
    }

    _handleWheel(e) {
        e.preventDefault();

        const pos = this._getCanvasPosition(e);
        const delta = e.deltaY > 0 ? 0.9 : 1.1;

        // Zoom um Mausposition
        const newScale = Math.max(0.1, Math.min(5, this.scale * delta));

        // Offset anpassen um auf Mausposition zu zoomen
        this.offsetX = pos.x - (pos.x - this.offsetX) * (newScale / this.scale);
        this.offsetY = pos.y - (pos.y - this.offsetY) * (newScale / this.scale);
        this.scale = newScale;

        this.render();
    }

    _handleTouchStart(e) {
        if (e.touches.length === 1) {
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('click', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this._handleClick(mouseEvent);
        }
    }

    _handleTouchMove(e) {
        // Pan mit einem Finger
        if (e.touches.length === 1 && this.isDragging) {
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this._handleMouseMove(mouseEvent);
        }
    }

    _handleTouchEnd(e) {
        this._handleMouseUp(e);
    }

    _getCanvasPosition(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    _canvasToImage(pos) {
        return {
            x: (pos.x - this.offsetX) / this.scale,
            y: (pos.y - this.offsetY) / this.scale
        };
    }

    _imageToCanvas(pos) {
        return {
            x: pos.x * this.scale + this.offsetX,
            y: pos.y * this.scale + this.offsetY
        };
    }

    _fitImageToCanvas() {
        if (!this.image) return;

        const scaleX = this.canvas.width / this.image.width;
        const scaleY = this.canvas.height / this.image.height;
        this.scale = Math.min(scaleX, scaleY) * 0.95;

        this.offsetX = (this.canvas.width - this.image.width * this.scale) / 2;
        this.offsetY = (this.canvas.height - this.image.height * this.scale) / 2;
    }

    _drawPoints() {
        for (let i = 0; i < this.points.length; i++) {
            const pt = this._imageToCanvas(this.points[i]);

            // Punkt
            this.ctx.beginPath();
            this.ctx.arc(pt.x, pt.y, this.options.pointRadius, 0, Math.PI * 2);
            this.ctx.fillStyle = this.hoveredPointIndex === i ? '#FF9500' : this.options.pointColor;
            this.ctx.fill();
            this.ctx.strokeStyle = '#FFF';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Nummer
            this.ctx.fillStyle = '#FFF';
            this.ctx.font = 'bold 12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(String(i + 1), pt.x, pt.y);
        }
    }

    _drawPolygon() {
        if (this.polygonPoints.length === 0) return;

        // Gefülltes Polygon (halbtransparent)
        if (this.polygonPoints.length >= 3) {
            this.ctx.beginPath();
            const first = this._imageToCanvas(this.polygonPoints[0]);
            this.ctx.moveTo(first.x, first.y);

            for (let i = 1; i < this.polygonPoints.length; i++) {
                const pt = this._imageToCanvas(this.polygonPoints[i]);
                this.ctx.lineTo(pt.x, pt.y);
            }

            this.ctx.closePath();
            this.ctx.fillStyle = 'rgba(0, 170, 255, 0.15)';
            this.ctx.fill();
        }

        // Linien
        this.ctx.beginPath();
        const first = this._imageToCanvas(this.polygonPoints[0]);
        this.ctx.moveTo(first.x, first.y);

        for (let i = 1; i < this.polygonPoints.length; i++) {
            const pt = this._imageToCanvas(this.polygonPoints[i]);
            this.ctx.lineTo(pt.x, pt.y);
        }

        if (this.polygonPoints.length >= 3) {
            this.ctx.closePath();
        }

        this.ctx.strokeStyle = this.options.polygonColor;
        this.ctx.lineWidth = this.options.lineWidth;
        this.ctx.stroke();

        // Punkte
        for (let i = 0; i < this.polygonPoints.length; i++) {
            const pt = this._imageToCanvas(this.polygonPoints[i]);

            this.ctx.beginPath();
            this.ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
            this.ctx.fillStyle = this.options.polygonColor;
            this.ctx.fill();
            this.ctx.strokeStyle = '#FFF';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }
    }

    /**
     * Zeichnet die "geraden Linien" - Hilfslinien die in der Realität gerade sind
     * aber im verzerrten Bild als Kurven erscheinen sollten
     */
    _drawStraightLines() {
        const colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF8800', '#88FF00'];

        // Fertige Linien zeichnen
        for (let i = 0; i < this.straightLines.length; i++) {
            const line = this.straightLines[i];
            if (line.length < 2) continue;

            const color = colors[i % colors.length];
            this._drawSingleLine(line, color, true);
        }

        // Aktuelle Linie (in Bearbeitung) zeichnen
        if (this.currentStraightLine.length > 0) {
            this._drawSingleLine(this.currentStraightLine, '#FF0000', false);
        }
    }

    /**
     * Zeichnet eine einzelne Linie mit Bezier-Kurve
     */
    _drawSingleLine(linePoints, color, showLabel) {
        if (linePoints.length < 1) return;

        // Punkte konvertieren
        const canvasPoints = linePoints.map(p => this._imageToCanvas(p));

        this.ctx.beginPath();
        this.ctx.moveTo(canvasPoints[0].x, canvasPoints[0].y);

        if (canvasPoints.length === 2) {
            // Einfache Linie bei 2 Punkten
            this.ctx.lineTo(canvasPoints[1].x, canvasPoints[1].y);
        } else if (canvasPoints.length >= 3) {
            // Spline durch alle Punkte für glatte Kurve
            for (let i = 1; i < canvasPoints.length - 1; i++) {
                const xc = (canvasPoints[i].x + canvasPoints[i + 1].x) / 2;
                const yc = (canvasPoints[i].y + canvasPoints[i + 1].y) / 2;
                this.ctx.quadraticCurveTo(canvasPoints[i].x, canvasPoints[i].y, xc, yc);
            }
            // Letztes Segment
            const last = canvasPoints[canvasPoints.length - 1];
            const secondLast = canvasPoints[canvasPoints.length - 2];
            this.ctx.quadraticCurveTo(secondLast.x, secondLast.y, last.x, last.y);
        }

        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.stroke();

        // Punkte auf der Linie zeichnen
        for (let i = 0; i < canvasPoints.length; i++) {
            const pt = canvasPoints[i];

            this.ctx.beginPath();
            this.ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            this.ctx.strokeStyle = '#FFF';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }

        // Label für fertige Linien
        if (showLabel && canvasPoints.length > 0) {
            const firstPt = canvasPoints[0];
            this.ctx.fillStyle = color;
            this.ctx.font = 'bold 14px Arial';
            this.ctx.textAlign = 'left';
            this.ctx.textBaseline = 'bottom';
            this.ctx.fillText('Gerade ' + (this.straightLines.indexOf(linePoints) + 1), firstPt.x + 8, firstPt.y - 5);
        }
    }

    // === Straight Lines Public Methods ===

    /**
     * Aktiviert den Modus zum Zeichnen von geraden Linien
     */
    setStraightLineMode(enabled) {
        this.straightLineMode = enabled;
        if (enabled) {
            this.options.mode = 'straightLine';
            this.canvas.style.cursor = 'crosshair';
        } else {
            this.options.mode = 'points';
        }
    }

    /**
     * Fügt einen Punkt zur aktuellen Linie hinzu
     */
    addStraightLinePoint(imagePos) {
        this.currentStraightLine.push(imagePos);
        this.render();
    }

    /**
     * Schliesst die aktuelle Linie ab und startet eine neue
     */
    finishStraightLine() {
        if (this.currentStraightLine.length >= 2) {
            this.straightLines.push([...this.currentStraightLine]);
        }
        this.currentStraightLine = [];
        this.render();
    }

    /**
     * Entfernt die letzte Linie
     */
    removeLastStraightLine() {
        if (this.currentStraightLine.length > 0) {
            this.currentStraightLine = [];
        } else if (this.straightLines.length > 0) {
            this.straightLines.pop();
        }
        this.render();
    }

    /**
     * Löscht alle Linien
     */
    clearStraightLines() {
        this.straightLines = [];
        this.currentStraightLine = [];
        this.render();
    }

    /**
     * Gibt die Anzahl der Linien zurück
     */
    getStraightLineCount() {
        return this.straightLines.length + (this.currentStraightLine.length >= 2 ? 1 : 0);
    }

    /**
     * Gibt alle Linien zurück (für Speicherung)
     */
    getStraightLines() {
        const lines = this.straightLines.map(line => line.map(p => [p.x, p.y]));
        if (this.currentStraightLine.length >= 2) {
            lines.push(this.currentStraightLine.map(p => [p.x, p.y]));
        }
        return lines;
    }

    /**
     * Setzt die Linien (für Wiederherstellung)
     */
    setStraightLines(lines) {
        this.straightLines = lines.map(line => line.map(p => ({ x: p[0], y: p[1] })));
        this.currentStraightLine = [];
        this.render();
    }

    _saveHistory() {
        // History nach aktuellem Index abschneiden
        this.history = this.history.slice(0, this.historyIndex + 1);

        // Neuen State speichern
        this.history.push({
            points: JSON.parse(JSON.stringify(this.points)),
            polygonPoints: JSON.parse(JSON.stringify(this.polygonPoints))
        });

        this.historyIndex = this.history.length - 1;

        // Max 50 History-Einträge
        if (this.history.length > 50) {
            this.history.shift();
            this.historyIndex--;
        }
    }

    _restoreFromHistory() {
        if (this.historyIndex >= 0 && this.historyIndex < this.history.length) {
            const state = this.history[this.historyIndex];
            this.points = JSON.parse(JSON.stringify(state.points));
            this.polygonPoints = JSON.parse(JSON.stringify(state.polygonPoints));
        }
    }
}


/**
 * Spielfeld-Canvas für Referenzpunkt-Auswahl
 * Verwendet das offizielle IFF Floorball-Spielfeld SVG als Hintergrund
 */
class FieldCanvas {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        this.fieldLength = options.fieldLength || 40; // Meter
        this.fieldWidth = options.fieldWidth || 20;   // Meter

        this.points = [];
        this.maxPoints = options.maxPoints || null;

        this.options = {
            pointColor: options.pointColor || '#FF6B00',
            onPointAdded: options.onPointAdded || null,
            onPointRemoved: options.onPointRemoved || null,
            svgUrl: options.svgUrl || '/static/images/Floorball_rink1.svg',
        };

        // SVG Hintergrundbild
        this.fieldImage = null;
        this.fieldImageLoaded = false;

        // SVG analysiert: Gesamtgröße 720x410px
        // SVG Transform: translate(-18.285715,-278.14789)
        // Um SVG-Koordinaten zu Canvas-Pixel zu konvertieren:
        // canvasX = svgX - 18.29, canvasY = svgY - 278.15
        //
        // Spielfeld-Rechteck (rect2167):
        //   SVG: x=23.57, y=284.86 → Canvas: x=5.28, y=6.71
        //   width=708.93, height=362.14
        this.svgField = {
            x: 5.28,
            y: 6.71,
            width: 708.93,
            height: 362.14
        };

        // SVG-Padding für das Zeichnen
        this.svgPadding = {
            left: this.svgField.x / 720,
            right: (720 - this.svgField.x - this.svgField.width) / 720,
            top: this.svgField.y / 410,
            bottom: (410 - this.svgField.y - this.svgField.height) / 410
        };

        // Bullypunkt-Positionen aus dem SVG (Canvas-Pixel nach Transform)
        // Berechnet aus den SVG path Elementen (Kreuz-Mittelpunkte)
        // SVG Transform: translate(-18.285715,-278.14789)
        // canvas_coord = svg_coord + transform
        //
        // Bully oben-links: Canvas (68.3, 57.0)
        // Bully unten-links: Canvas (68.3, 328.1)
        // Bully oben-rechts: Canvas (652.2, 56.3)
        // Bully unten-rechts: Canvas (652.3, 328.1)
        // Bully Mitte oben: Canvas (351.3, 56.2)
        // Bully Mitte unten: Canvas (351.1, 328.3)
        // Bully Mitte: Canvas (352.0, 188.2)
        const svgBullys = {
            topLeft:     { x: (68.3 - this.svgField.x) / this.svgField.width,     y: (57.0 - this.svgField.y) / this.svgField.height },
            bottomLeft:  { x: (68.3 - this.svgField.x) / this.svgField.width,     y: (328.1 - this.svgField.y) / this.svgField.height },
            topRight:    { x: (652.2 - this.svgField.x) / this.svgField.width,    y: (56.3 - this.svgField.y) / this.svgField.height },
            bottomRight: { x: (652.3 - this.svgField.x) / this.svgField.width,    y: (328.1 - this.svgField.y) / this.svgField.height },
            midTop:      { x: (351.3 - this.svgField.x) / this.svgField.width,    y: (56.2 - this.svgField.y) / this.svgField.height },
            midBottom:   { x: (351.1 - this.svgField.x) / this.svgField.width,    y: (328.3 - this.svgField.y) / this.svgField.height },
            center:      { x: (352.0 - this.svgField.x) / this.svgField.width,    y: (188.2 - this.svgField.y) / this.svgField.height }
        };

        // Torraum-Rechteck aus SVG (rect5084 für links, rect7030 gespiegelt für rechts)
        // Links: Canvas x=53.5, y=137.3, w=73.7, h=95.9
        // Rechts: Canvas x=595.0, y=137.2, w=73.7, h=95.9
        const svgGoalArea = {
            leftX: (53.5 - this.svgField.x) / this.svgField.width,
            leftWidth: 73.7 / this.svgField.width,
            rightX: (595.0 - this.svgField.x) / this.svgField.width,
            topY: (137.3 - this.svgField.y) / this.svgField.height,
            height: 95.9 / this.svgField.height
        };

        // Konvertiere SVG-Anteile zu Spielfeld-Metern (für visuelle Position)
        // UND speichere echte IFF-Koordinaten (für Homography)
        const toFieldX = (ratio) => ratio * this.fieldLength;
        const toFieldY = (ratio) => (1 - ratio) * this.fieldWidth;  // SVG y ist invertiert

        // IFF-konforme echte Koordinaten
        const iffBullyFromSideWall = 1.5;
        const iffBullyFromEndWall = 3.5;
        const iffGoalAreaRadius = 4;
        const iffGoalDistance = 2.85;

        const midX = this.fieldLength / 2;
        const midY = this.fieldWidth / 2;

        // SVG-Mittellinie Position (aus path7038 im SVG)
        // Die SVG-Mittellinie ist bei Canvas X = 351.8, was ~0.4888 des Spielfelds ist
        const svgMidlineRatio = (351.8 - this.svgField.x) / this.svgField.width;

        // Referenzpunkte mit SVG-Position (für Anzeige) und IFF-Koordinaten (für Homography)
        // ALLE Punkte haben svgX/svgY für die visuelle Darstellung auf dem SVG
        // lineIds: Array von Linien-IDs auf denen dieser Punkt liegt (für Fisheye-Schätzung)
        //   - horizontal: h_top, h_bottom, h_torraum_links_oben, h_torraum_links_unten, etc.
        //   - vertikal: v_left, v_right, v_middle, v_torraum_links_aussen, etc.
        this.referencePoints = [
            // === BULLYPUNKTE (7 Stück) ===
            // svgX/svgY = Position auf dem SVG-Hintergrund
            // x/y = IFF-konforme Koordinaten für Homography
            { name: 'Bully Mitte',
              svgX: toFieldX(svgBullys.center.x), svgY: toFieldY(svgBullys.center.y),
              x: midX, y: midY, isBully: true, lineIds: [] },
            { name: 'Bully oben-links',
              svgX: toFieldX(svgBullys.topLeft.x), svgY: toFieldY(svgBullys.topLeft.y),
              x: iffBullyFromEndWall, y: this.fieldWidth - iffBullyFromSideWall, isBully: true, lineIds: [] },
            { name: 'Bully unten-links',
              svgX: toFieldX(svgBullys.bottomLeft.x), svgY: toFieldY(svgBullys.bottomLeft.y),
              x: iffBullyFromEndWall, y: iffBullyFromSideWall, isBully: true, lineIds: [] },
            { name: 'Bully oben-rechts',
              svgX: toFieldX(svgBullys.topRight.x), svgY: toFieldY(svgBullys.topRight.y),
              x: this.fieldLength - iffBullyFromEndWall, y: this.fieldWidth - iffBullyFromSideWall, isBully: true, lineIds: [] },
            { name: 'Bully unten-rechts',
              svgX: toFieldX(svgBullys.bottomRight.x), svgY: toFieldY(svgBullys.bottomRight.y),
              x: this.fieldLength - iffBullyFromEndWall, y: iffBullyFromSideWall, isBully: true, lineIds: [] },
            { name: 'Bully Mitte oben',
              svgX: toFieldX(svgBullys.midTop.x), svgY: toFieldY(svgBullys.midTop.y),
              x: midX, y: this.fieldWidth - iffBullyFromSideWall, isBully: true, lineIds: ['v_middle'] },
            { name: 'Bully Mitte unten',
              svgX: toFieldX(svgBullys.midBottom.x), svgY: toFieldY(svgBullys.midBottom.y),
              x: midX, y: iffBullyFromSideWall, isBully: true, lineIds: ['v_middle'] },

            // === Spielfeld-Ecken ===
            // svgX/svgY = SVG-Spielfeldrand (0 oder 1 als Anteil)
            { name: 'Ecke oben-links',
              svgX: 0, svgY: this.fieldWidth,
              x: 0, y: this.fieldWidth, lineIds: ['h_top', 'v_left'] },
            { name: 'Ecke oben-rechts',
              svgX: this.fieldLength, svgY: this.fieldWidth,
              x: this.fieldLength, y: this.fieldWidth, lineIds: ['h_top', 'v_right'] },
            { name: 'Ecke unten-links',
              svgX: 0, svgY: 0,
              x: 0, y: 0, lineIds: ['h_bottom', 'v_left'] },
            { name: 'Ecke unten-rechts',
              svgX: this.fieldLength, svgY: 0,
              x: this.fieldLength, y: 0, lineIds: ['h_bottom', 'v_right'] },

            // === Mittellinie (SVG-Mittellinie verwenden!) ===
            { name: 'Mitte oben (Bande)',
              svgX: toFieldX(svgMidlineRatio), svgY: this.fieldWidth,
              x: midX, y: this.fieldWidth, lineIds: ['h_top', 'v_middle'] },
            { name: 'Mitte unten (Bande)',
              svgX: toFieldX(svgMidlineRatio), svgY: 0,
              x: midX, y: 0, lineIds: ['h_bottom', 'v_middle'] },

            // === Seitenmitte ===
            { name: 'Mitte links',
              svgX: 0, svgY: midY,
              x: 0, y: midY, lineIds: ['v_left'] },
            { name: 'Mitte rechts',
              svgX: this.fieldLength, svgY: midY,
              x: this.fieldLength, y: midY, lineIds: ['v_right'] },

            // === Torraum-Ecken (4 pro Seite, Rechteck im SVG) ===
            // Linker Torraum
            { name: 'Torraum links oben-aussen',
              svgX: toFieldX(svgGoalArea.leftX), svgY: toFieldY(svgGoalArea.topY),
              x: iffGoalDistance, y: midY + iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_l_top', 'v_torraum_l_outer'] },
            { name: 'Torraum links unten-aussen',
              svgX: toFieldX(svgGoalArea.leftX), svgY: toFieldY(svgGoalArea.topY + svgGoalArea.height),
              x: iffGoalDistance, y: midY - iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_l_bottom', 'v_torraum_l_outer'] },
            { name: 'Torraum links oben-innen',
              svgX: toFieldX(svgGoalArea.leftX + svgGoalArea.leftWidth), svgY: toFieldY(svgGoalArea.topY),
              x: iffGoalDistance + iffGoalAreaRadius, y: midY + iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_l_top', 'v_torraum_l_inner'] },
            { name: 'Torraum links unten-innen',
              svgX: toFieldX(svgGoalArea.leftX + svgGoalArea.leftWidth), svgY: toFieldY(svgGoalArea.topY + svgGoalArea.height),
              x: iffGoalDistance + iffGoalAreaRadius, y: midY - iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_l_bottom', 'v_torraum_l_inner'] },

            // Rechter Torraum
            { name: 'Torraum rechts oben-aussen',
              svgX: toFieldX(svgGoalArea.rightX + svgGoalArea.leftWidth), svgY: toFieldY(svgGoalArea.topY),
              x: this.fieldLength - iffGoalDistance, y: midY + iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_r_top', 'v_torraum_r_outer'] },
            { name: 'Torraum rechts unten-aussen',
              svgX: toFieldX(svgGoalArea.rightX + svgGoalArea.leftWidth), svgY: toFieldY(svgGoalArea.topY + svgGoalArea.height),
              x: this.fieldLength - iffGoalDistance, y: midY - iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_r_bottom', 'v_torraum_r_outer'] },
            { name: 'Torraum rechts oben-innen',
              svgX: toFieldX(svgGoalArea.rightX), svgY: toFieldY(svgGoalArea.topY),
              x: this.fieldLength - iffGoalDistance - iffGoalAreaRadius, y: midY + iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_r_top', 'v_torraum_r_inner'] },
            { name: 'Torraum rechts unten-innen',
              svgX: toFieldX(svgGoalArea.rightX), svgY: toFieldY(svgGoalArea.topY + svgGoalArea.height),
              x: this.fieldLength - iffGoalDistance - iffGoalAreaRadius, y: midY - iffGoalAreaRadius, isTorraum: true,
              lineIds: ['h_torraum_r_bottom', 'v_torraum_r_inner'] },
        ];

        // Definition welche Linien horizontal/vertikal sind (für Fisheye-Schätzung)
        this.lineDefinitions = {
            // Horizontale Linien (y ist konstant in Realität)
            'h_top': { type: 'horizontal', description: 'Obere Bande' },
            'h_bottom': { type: 'horizontal', description: 'Untere Bande' },
            'h_torraum_l_top': { type: 'horizontal', description: 'Torraum links oben' },
            'h_torraum_l_bottom': { type: 'horizontal', description: 'Torraum links unten' },
            'h_torraum_r_top': { type: 'horizontal', description: 'Torraum rechts oben' },
            'h_torraum_r_bottom': { type: 'horizontal', description: 'Torraum rechts unten' },
            // Vertikale Linien (x ist konstant in Realität)
            'v_left': { type: 'vertical', description: 'Linke Bande' },
            'v_right': { type: 'vertical', description: 'Rechte Bande' },
            'v_middle': { type: 'vertical', description: 'Mittellinie' },
            'v_torraum_l_outer': { type: 'vertical', description: 'Torraum links aussen' },
            'v_torraum_l_inner': { type: 'vertical', description: 'Torraum links innen' },
            'v_torraum_r_outer': { type: 'vertical', description: 'Torraum rechts aussen' },
            'v_torraum_r_inner': { type: 'vertical', description: 'Torraum rechts innen' },
        };

        this.hoveredRef = null;
        this.scale = 1;
        this.padding = 40;
        this.rotation = 0;  // 0, 90, 180, 270 Grad

        this._calculateScale();
        this._setupEventListeners();

        // SVG-Hintergrund laden
        this._loadFieldImage();
    }

    /**
     * Setzt die Rotation des Spielfelds (für Hintertorkameras)
     * @param {number} degrees - 0, 90, 180 oder 270
     */
    setRotation(degrees) {
        this.rotation = degrees % 360;
        this._calculateScale();
        this.render();
    }

    /**
     * Lädt das Spielfeld-SVG als Hintergrundbild
     */
    _loadFieldImage() {
        this.fieldImage = new Image();
        this.fieldImage.onload = () => {
            this.fieldImageLoaded = true;
            this.render();
        };
        this.fieldImage.onerror = () => {
            console.warn('Spielfeld-SVG konnte nicht geladen werden, verwende programmatische Zeichnung');
            this.fieldImageLoaded = false;
            this.render();
        };
        this.fieldImage.src = this.options.svgUrl;
    }

    setPoints(points) {
        this.points = points.map(p => ({ x: p[0], y: p[1] }));
        this.render();
    }

    getPoints() {
        return this.points.map(p => [p.x, p.y]);
    }

    clearPoints() {
        this.points = [];
        this.render();
    }

    removeLastPoint() {
        if (this.points.length > 0) {
            this.points.pop();
            this.render();
            if (this.options.onPointRemoved) {
                this.options.onPointRemoved(this.points.length);
            }
        }
    }

    /**
     * Punkt an einem bestimmten Index entfernen
     */
    removePointAtIndex(index) {
        if (index >= 0 && index < this.points.length) {
            this.points.splice(index, 1);
            this.render();
            if (this.options.onPointRemoved) {
                this.options.onPointRemoved(this.points.length);
            }
        }
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this._drawField();
        this._drawReferencePoints();
        this._drawSelectedPoints();
    }

    _calculateScale() {
        const availableWidth = this.canvas.width - 2 * this.padding;
        const availableHeight = this.canvas.height - 2 * this.padding;

        // Bei 90° oder 270° Rotation: Breite und Höhe tauschen
        const isRotated = (this.rotation === 90 || this.rotation === 270);
        const effectiveLength = isRotated ? this.fieldWidth : this.fieldLength;
        const effectiveWidth = isRotated ? this.fieldLength : this.fieldWidth;

        this.scale = Math.min(
            availableWidth / effectiveLength,
            availableHeight / effectiveWidth
        );
    }

    _fieldToCanvas(x, y) {
        // Erst die Rotation anwenden, dann zu Canvas-Koordinaten konvertieren
        let rotX = x, rotY = y;

        switch (this.rotation) {
            case 90:
                // 90° im Uhrzeigersinn: (x, y) → (y, fieldLength - x)
                rotX = y;
                rotY = this.fieldLength - x;
                break;
            case 180:
                // 180°: (x, y) → (fieldLength - x, fieldWidth - y)
                rotX = this.fieldLength - x;
                rotY = this.fieldWidth - y;
                break;
            case 270:
                // 270° im Uhrzeigersinn: (x, y) → (fieldWidth - y, x)
                rotX = this.fieldWidth - y;
                rotY = x;
                break;
            // 0°: keine Änderung
        }

        // Bei 90° oder 270° sind die Dimensionen getauscht
        const isRotated = (this.rotation === 90 || this.rotation === 270);
        const effectiveLength = isRotated ? this.fieldWidth : this.fieldLength;
        const effectiveWidth = isRotated ? this.fieldLength : this.fieldWidth;

        return {
            x: this.padding + rotX * this.scale,
            y: this.canvas.height - this.padding - rotY * this.scale
        };
    }

    _canvasToField(canvasX, canvasY) {
        // Erst zu Spielfeld-Koordinaten (rotiert), dann Rotation rückgängig machen
        const isRotated = (this.rotation === 90 || this.rotation === 270);
        const effectiveLength = isRotated ? this.fieldWidth : this.fieldLength;
        const effectiveWidth = isRotated ? this.fieldLength : this.fieldWidth;

        const rotX = (canvasX - this.padding) / this.scale;
        const rotY = (this.canvas.height - this.padding - canvasY) / this.scale;

        // Rotation rückgängig machen
        let x = rotX, y = rotY;

        switch (this.rotation) {
            case 90:
                // Inverse von 90°: (rotX, rotY) → (fieldLength - rotY, rotX)
                x = this.fieldLength - rotY;
                y = rotX;
                break;
            case 180:
                // Inverse von 180°: (rotX, rotY) → (fieldLength - rotX, fieldWidth - rotY)
                x = this.fieldLength - rotX;
                y = this.fieldWidth - rotY;
                break;
            case 270:
                // Inverse von 270°: (rotX, rotY) → (rotY, fieldWidth - rotX)
                x = rotY;
                y = this.fieldWidth - rotX;
                break;
            // 0°: keine Änderung
        }

        return { x, y };
    }

    _setupEventListeners() {
        this.canvas.addEventListener('click', (e) => this._handleClick(e));
        this.canvas.addEventListener('mousemove', (e) => this._handleMouseMove(e));
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.removeLastPoint();
        });
    }

    _handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        // Prüfen ob auf Referenzpunkt geklickt
        // Verwende svgX/svgY für die visuelle Position, x/y für die IFF-Koordinaten
        for (const ref of this.referencePoints) {
            const displayX = ref.svgX !== undefined ? ref.svgX : ref.x;
            const displayY = ref.svgY !== undefined ? ref.svgY : ref.y;
            const pt = this._fieldToCanvas(displayX, displayY);
            const dist = Math.sqrt(Math.pow(canvasX - pt.x, 2) + Math.pow(canvasY - pt.y, 2));

            if (dist < 12) {
                if (this.maxPoints && this.points.length >= this.maxPoints) {
                    return;
                }

                // Speichere IFF-konforme Koordinaten (x/y), nicht SVG-Koordinaten
                this.points.push({ x: ref.x, y: ref.y });
                this.render();

                if (this.options.onPointAdded) {
                    this.options.onPointAdded(this.points.length, { x: ref.x, y: ref.y }, ref.name);
                }
                return;
            }
        }

        // Freie Punkt-Platzierung auf dem Feld
        const fieldPos = this._canvasToField(canvasX, canvasY);

        // Nur innerhalb des Feldes
        if (fieldPos.x >= 0 && fieldPos.x <= this.fieldLength &&
            fieldPos.y >= 0 && fieldPos.y <= this.fieldWidth) {

            if (this.maxPoints && this.points.length >= this.maxPoints) {
                return;
            }

            // Auf 0.5m runden
            fieldPos.x = Math.round(fieldPos.x * 2) / 2;
            fieldPos.y = Math.round(fieldPos.y * 2) / 2;

            this.points.push(fieldPos);
            this.render();

            if (this.options.onPointAdded) {
                this.options.onPointAdded(this.points.length, fieldPos, null);
            }
        }
    }

    _handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        this.hoveredRef = null;

        for (const ref of this.referencePoints) {
            // Verwende svgX/svgY für die visuelle Position
            const displayX = ref.svgX !== undefined ? ref.svgX : ref.x;
            const displayY = ref.svgY !== undefined ? ref.svgY : ref.y;
            const pt = this._fieldToCanvas(displayX, displayY);
            const dist = Math.sqrt(Math.pow(canvasX - pt.x, 2) + Math.pow(canvasY - pt.y, 2));

            if (dist < 12) {
                this.hoveredRef = ref;
                break;
            }
        }

        this.canvas.style.cursor = this.hoveredRef ? 'pointer' : 'crosshair';
        this.render();
    }

    _drawField() {
        const ctx = this.ctx;
        const topLeft = this._fieldToCanvas(0, this.fieldWidth);
        const bottomRight = this._fieldToCanvas(this.fieldLength, 0);
        const fieldWidthPx = bottomRight.x - topLeft.x;
        const fieldHeightPx = bottomRight.y - topLeft.y;

        // Wenn SVG geladen ist, dieses als Hintergrund verwenden
        if (this.fieldImageLoaded && this.fieldImage) {
            // Das SVG hat einen Rand mit Dimensionspfeilen
            // SVG Gesamtgröße: 720x410px
            // Spielfeld innerhalb SVG: x=5.28, y=6.71, w=708.93, h=362.14
            //
            // Wir müssen das SVG so positionieren und skalieren, dass das
            // innere Spielfeld-Rechteck genau auf unseren Canvas-Bereich passt

            ctx.save();

            // Bei Rotation um den Mittelpunkt des Spielfelds rotieren
            if (this.rotation !== 0) {
                const centerX = (topLeft.x + bottomRight.x) / 2;
                const centerY = (topLeft.y + bottomRight.y) / 2;
                ctx.translate(centerX, centerY);
                ctx.rotate(this.rotation * Math.PI / 180);
                ctx.translate(-centerX, -centerY);
            }

            // Berechne Skalierung: Das innere Spielfeld (708.93 x 362.14) soll
            // auf fieldWidthPx x fieldHeightPx passen
            const scaleX = Math.abs(fieldWidthPx) / this.svgField.width;
            const scaleY = Math.abs(fieldHeightPx) / this.svgField.height;

            // Das SVG muss so positioniert werden, dass das innere Spielfeld
            // (das bei svgField.x, svgField.y beginnt) genau bei topLeft liegt
            // SVG-Position = topLeft - (svgField-Offset * scale)
            const svgX = topLeft.x - this.svgField.x * scaleX;
            const svgY = topLeft.y - this.svgField.y * scaleY;

            // SVG-Gesamtgröße skaliert
            const svgWidth = 720 * scaleX;
            const svgHeight = 410 * scaleY;

            ctx.drawImage(
                this.fieldImage,
                svgX,
                svgY,
                svgWidth,
                svgHeight
            );

            ctx.restore();
        } else {
            // Fallback: Einfacher Hintergrund wenn SVG nicht geladen
            ctx.fillStyle = '#87AE73';  // Grüner Hallenboden
            ctx.fillRect(topLeft.x, topLeft.y, fieldWidthPx, fieldHeightPx);

            // Spielfeld-Rand
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 4;
            ctx.strokeRect(topLeft.x, topLeft.y, fieldWidthPx, fieldHeightPx);

            // Mittellinie
            ctx.strokeStyle = '#C00';
            ctx.lineWidth = 2;
            const midTop = this._fieldToCanvas(this.fieldLength / 2, this.fieldWidth);
            const midBottom = this._fieldToCanvas(this.fieldLength / 2, 0);
            ctx.beginPath();
            ctx.moveTo(midTop.x, midTop.y);
            ctx.lineTo(midBottom.x, midBottom.y);
            ctx.stroke();
        }

        // Dimensionen-Beschriftung nur anzeigen wenn SVG nicht geladen
        // (das SVG hat bereits 40m und 20m Labels)
        if (!this.fieldImageLoaded) {
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';

            const lengthLabel = this._fieldToCanvas(this.fieldLength / 2, -1.5);
            ctx.fillText(`${this.fieldLength}m`, lengthLabel.x, lengthLabel.y);

            ctx.save();
            ctx.translate(this.padding - 25, this.canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(`${this.fieldWidth}m`, 0, 0);
            ctx.restore();
        }
    }

    _drawReferencePoints() {
        const ctx = this.ctx;

        for (const ref of this.referencePoints) {
            // Verwende svgX/svgY wenn vorhanden (für SVG-Ausrichtung), sonst x/y
            const displayX = ref.svgX !== undefined ? ref.svgX : ref.x;
            const displayY = ref.svgY !== undefined ? ref.svgY : ref.y;
            const pt = this._fieldToCanvas(displayX, displayY);
            const isHovered = this.hoveredRef === ref;
            const isBully = ref.isBully;
            const isTorraum = ref.name && ref.name.includes('Torraum');

            // Bullypunkte: Kreuz zeichnen (wie im offiziellen Spielfeld)
            if (isBully) {
                const crossSize = isHovered ? 14 : 10;
                ctx.strokeStyle = isHovered ? '#FF0' : '#0A0';
                ctx.lineWidth = 3;

                // Horizontale Linie
                ctx.beginPath();
                ctx.moveTo(pt.x - crossSize, pt.y);
                ctx.lineTo(pt.x + crossSize, pt.y);
                ctx.stroke();

                // Vertikale Linie
                ctx.beginPath();
                ctx.moveTo(pt.x, pt.y - crossSize);
                ctx.lineTo(pt.x, pt.y + crossSize);
                ctx.stroke();

                // Kleiner Punkt in der Mitte
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
                ctx.fillStyle = isHovered ? '#FF0' : '#0A0';
                ctx.fill();
            } else if (isTorraum) {
                // Torraum-Ecken: Kleine Quadrate
                const size = isHovered ? 10 : 7;
                ctx.fillStyle = isHovered ? 'rgba(255, 255, 0, 0.9)' : 'rgba(0, 150, 255, 0.7)';
                ctx.fillRect(pt.x - size/2, pt.y - size/2, size, size);
                ctx.strokeStyle = '#00F';
                ctx.lineWidth = 2;
                ctx.strokeRect(pt.x - size/2, pt.y - size/2, size, size);
            } else {
                // Andere Referenzpunkte als Kreise
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, isHovered ? 12 : 8, 0, Math.PI * 2);
                ctx.fillStyle = isHovered ? 'rgba(255, 255, 0, 0.9)' : 'rgba(255, 255, 255, 0.5)';
                ctx.fill();
                ctx.strokeStyle = '#FFF';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // Tooltip bei Hover
            if (isHovered) {
                ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
                ctx.font = '12px Arial';
                const text = `${ref.name} (${ref.x.toFixed(1)}, ${ref.y.toFixed(1)})m`;
                const textWidth = ctx.measureText(text).width;
                ctx.fillRect(pt.x - textWidth / 2 - 5, pt.y - 35, textWidth + 10, 25);

                ctx.fillStyle = '#FFF';
                ctx.textAlign = 'center';
                ctx.fillText(text, pt.x, pt.y - 18);
            }
        }
    }

    _drawSelectedPoints() {
        const ctx = this.ctx;

        for (let i = 0; i < this.points.length; i++) {
            const pt = this._fieldToCanvas(this.points[i].x, this.points[i].y);

            // Punkt
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = this.options.pointColor;
            ctx.fill();
            ctx.strokeStyle = '#FFF';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Nummer
            ctx.fillStyle = '#FFF';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(i + 1), pt.x, pt.y);

            // Koordinaten
            ctx.fillStyle = '#FFF';
            ctx.font = '10px Arial';
            ctx.fillText(`(${this.points[i].x}, ${this.points[i].y})`, pt.x, pt.y + 20);
        }
    }
}
