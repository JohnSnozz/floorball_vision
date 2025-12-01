"""
Analysis Routes - Video-Analyse mit Tracking
"""
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, abort, send_file
from src.web.extensions import db
from src.db.models import Video, Calibration

analysis_bp = Blueprint("analysis", __name__, url_prefix="/analysis")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@analysis_bp.route("/")
def index():
    """Analyse-Übersicht - Videos mit Kalibrierung."""
    # Nur Videos mit aktiver Kalibrierung anzeigen
    videos_with_calibration = []

    videos = db.session.query(Video).filter(Video.status == "ready").all()
    for video in videos:
        calibration = db.session.query(Calibration).filter_by(
            video_id=video.id,
            is_active=True
        ).first()
        if calibration:
            videos_with_calibration.append({
                "video": video,
                "calibration": calibration
            })

    return render_template(
        "analysis/index.html",
        videos=videos_with_calibration
    )


@analysis_bp.route("/video/<video_id>")
def video_analysis(video_id):
    """Video-Analyse-Seite mit Snippet-Auswahl."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    # Aktive Kalibrierung für dieses Video
    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    # Alle Kalibrierungen für dieses Video (für Dropdown-Auswahl)
    all_calibrations = db.session.query(Calibration).filter_by(
        video_id=video.id
    ).order_by(Calibration.created_at.desc()).all()

    # Bestehende Snippets laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    snippets = []
    if snippets_file.exists():
        with open(snippets_file, "r") as f:
            snippets = json.load(f)

    return render_template(
        "analysis/video.html",
        video=video,
        calibration=calibration,
        all_calibrations=all_calibrations,
        snippets=snippets
    )


@analysis_bp.route("/video/<video_id>/review/<snippet_id>")
def review_tracking(video_id, snippet_id):
    """Tracking-Review-Seite für ein Snippet."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    # Snippet-Daten laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        abort(404)

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        abort(404)

    # Tracking-Daten laden
    tracking_file = PROJECT_ROOT / "data" / "analysis" / video_id / "tracking" / f"{snippet_id}.json"
    tracking_data = None
    if tracking_file.exists():
        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)

    return render_template(
        "analysis/review.html",
        video=video,
        snippet=snippet,
        tracking_data=tracking_data
    )


# === API Endpoints ===

@analysis_bp.route("/api/videos/<video_id>/snippets", methods=["GET"])
def api_get_snippets(video_id):
    """Lädt alle Snippets für ein Video."""
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"

    if not snippets_file.exists():
        return jsonify({"snippets": []})

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    return jsonify({"snippets": snippets})


@analysis_bp.route("/api/videos/<video_id>/snippets", methods=["POST"])
def api_save_snippets(video_id):
    """Speichert Snippets für ein Video."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    snippets = data.get("snippets", [])

    # Validierung: max 5 Snippets, 10-20 Sekunden
    if len(snippets) > 5:
        return jsonify({"error": "Maximal 5 Snippets erlaubt"}), 400

    for snippet in snippets:
        duration = snippet.get("end", 0) - snippet.get("start", 0)
        if duration < 5 or duration > 30:
            return jsonify({"error": f"Snippet-Dauer muss zwischen 5 und 30 Sekunden liegen (aktuell: {duration:.1f}s)"}), 400

    # Verzeichnis erstellen
    analysis_dir = PROJECT_ROOT / "data" / "analysis" / video_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # IDs generieren falls nicht vorhanden
    for i, snippet in enumerate(snippets):
        if not snippet.get("id"):
            snippet["id"] = f"snippet_{i+1}_{int(datetime.now().timestamp())}"

    # Speichern
    snippets_file = analysis_dir / "snippets.json"
    with open(snippets_file, "w") as f:
        json.dump(snippets, f, indent=2)

    return jsonify({"success": True, "snippets": snippets})


@analysis_bp.route("/api/videos/<video_id>/boundary", methods=["GET"])
def api_get_boundary(video_id):
    """Lädt das Spielfeld-Boundary-Polygon."""
    boundary_file = PROJECT_ROOT / "data" / "analysis" / video_id / "boundary.json"

    if not boundary_file.exists():
        return jsonify({"boundary_polygon": []})

    with open(boundary_file, "r") as f:
        data = json.load(f)

    return jsonify(data)


@analysis_bp.route("/api/videos/<video_id>/boundary", methods=["POST"])
def api_save_boundary(video_id):
    """Speichert das Spielfeld-Boundary-Polygon."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    boundary_polygon = data.get("boundary_polygon", [])

    # Validierung: mindestens 3 Punkte
    if len(boundary_polygon) < 3:
        return jsonify({"error": "Mindestens 3 Punkte erforderlich"}), 400

    # Verzeichnis erstellen
    analysis_dir = PROJECT_ROOT / "data" / "analysis" / video_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Speichern
    boundary_file = analysis_dir / "boundary.json"
    with open(boundary_file, "w") as f:
        json.dump({"boundary_polygon": boundary_polygon}, f, indent=2)

    return jsonify({"success": True, "boundary_polygon": boundary_polygon})


@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/progress", methods=["GET"])
def api_get_tracking_progress(video_id, snippet_id):
    """Gibt den aktuellen Tracking-Progress zurück."""
    progress_file = PROJECT_ROOT / "data" / "analysis" / video_id / f"progress_{snippet_id}.json"

    if not progress_file.exists():
        return jsonify({"stage": "idle", "percent": 0, "detail": ""})

    try:
        with open(progress_file, "r") as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"stage": "idle", "percent": 0, "detail": ""})


@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/extract", methods=["POST"])
def api_extract_snippet(video_id, snippet_id):
    """Extrahiert ein Snippet mit ffmpeg."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Snippet-Daten laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        return jsonify({"error": "Keine Snippets gefunden"}), 404

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        return jsonify({"error": "Snippet nicht gefunden"}), 404

    # Pfade
    video_path = PROJECT_ROOT / video.file_path
    output_dir = PROJECT_ROOT / "data" / "analysis" / video_id / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{snippet_id}.mp4"

    # ffmpeg Befehl
    start_time = snippet["start"]
    duration = snippet["end"] - snippet["start"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return jsonify({"error": f"ffmpeg Fehler: {result.stderr[:500]}"}), 500

        # Snippet-Status aktualisieren
        snippet["extracted"] = True
        snippet["clip_path"] = str(output_path.relative_to(PROJECT_ROOT))

        with open(snippets_file, "w") as f:
            json.dump(snippets, f, indent=2)

        return jsonify({
            "success": True,
            "clip_path": snippet["clip_path"]
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "ffmpeg Timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def point_in_polygon(x, y, polygon):
    """Prüft ob ein Punkt innerhalb eines Polygons liegt (Ray-Casting)."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def load_clip_model():
    """Lädt das CLIP-Modell für Team-Zuweisung (singleton)."""
    if not hasattr(load_clip_model, "_model"):
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("Loading CLIP model for team assignment...")
            load_clip_model._model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            load_clip_model._processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"ERROR loading CLIP model: {e}")
            load_clip_model._model = None
            load_clip_model._processor = None
    return load_clip_model._model, load_clip_model._processor


def load_ocr_reader():
    """Lädt EasyOCR Reader für Rückennummer-Erkennung (singleton)."""
    if not hasattr(load_ocr_reader, "_reader"):
        try:
            import easyocr
            print("Loading EasyOCR for jersey number recognition...")
            # GPU kann Probleme machen, versuche ohne
            load_ocr_reader._reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR loaded successfully")
        except Exception as e:
            print(f"ERROR loading EasyOCR: {e}")
            load_ocr_reader._reader = None
    return load_ocr_reader._reader


def detect_jersey_number(frame, bbox, ocr_reader):
    """Erkennt Rückennummer auf einem Spieler-Crop."""
    import cv2
    import re

    x1, y1, x2, y2 = [int(c) for c in bbox]
    h = y2 - y1
    w = x2 - x1

    # Nur den oberen Teil (Rücken/Brust) des Spielers betrachten
    # Typischerweise ist die Nummer im oberen 60% der Box
    crop_y1 = y1
    crop_y2 = y1 + int(h * 0.6)
    crop_x1 = x1 + int(w * 0.1)  # Etwas Rand abschneiden
    crop_x2 = x2 - int(w * 0.1)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return None, 0.0

    player_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if player_crop.size == 0:
        return None, 0.0

    try:
        # OCR durchführen
        results = ocr_reader.readtext(player_crop, allowlist='0123456789')

        # Nur Zahlen 1-99 akzeptieren
        for (bbox_ocr, text, conf) in results:
            # Nur Ziffern extrahieren
            numbers = re.findall(r'\d+', text)
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if 1 <= num <= 99:
                        return num, float(conf)
                except ValueError:
                    continue

        return None, 0.0
    except Exception as e:
        return None, 0.0


def get_team_assignment(frame, bbox, team_config, clip_model, clip_processor):
    """Weist einer Detection ein Team zu basierend auf CLIP."""
    import cv2
    from PIL import Image
    import torch

    x1, y1, x2, y2 = [int(c) for c in bbox]

    # Crop player from frame
    player_img = frame[y1:y2, x1:x2]
    if player_img.size == 0:
        return "unknown", 0.0

    # Convert to PIL
    rgb_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    # Team descriptions
    classes = [
        team_config["team1"]["color"],
        team_config["team2"]["color"],
        team_config["referee"]["color"]
    ]

    # CLIP inference
    inputs = clip_processor(text=classes, images=pil_img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)[0].tolist()

    # Best match
    best_idx = probs.index(max(probs))
    best_prob = probs[best_idx]

    if best_idx == 0:
        return "team1", best_prob
    elif best_idx == 1:
        return "team2", best_prob
    else:
        return "referee", best_prob


@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/track", methods=["POST"])
def api_track_snippet(video_id, snippet_id):
    """Führt YOLO-Tracking auf einem Snippet aus mit Team-Zuweisung."""
    from ultralytics import YOLO
    import cv2
    import numpy as np

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Spielfeld-Polygon laden (primär aus boundary.json)
    field_polygon = None
    boundary_file = PROJECT_ROOT / "data" / "analysis" / video_id / "boundary.json"
    if boundary_file.exists():
        with open(boundary_file, "r") as f:
            boundary_data = json.load(f)
            field_polygon = boundary_data.get("boundary_polygon", [])
            if len(field_polygon) < 3:
                field_polygon = None

    # Fallback: aus Kalibrierung laden
    if field_polygon is None:
        calibration = db.session.query(Calibration).filter_by(
            video_id=video.id,
            is_active=True
        ).first()
        if calibration and calibration.calibration_data:
            step2 = calibration.calibration_data.get("step2_boundary", {})
            field_polygon = step2.get("boundary_polygon", [])
            if len(field_polygon) < 3:
                field_polygon = None

    # Snippet-Daten laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        return jsonify({"error": "Keine Snippets gefunden"}), 404

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        return jsonify({"error": "Snippet nicht gefunden"}), 404

    # Clip-Pfad prüfen
    clip_path = PROJECT_ROOT / snippet.get("clip_path", "")
    if not clip_path.exists():
        return jsonify({"error": "Clip nicht gefunden. Bitte zuerst extrahieren."}), 404

    # Request-Parameter
    data = request.get_json() or {}
    model_path = data.get("model_path", "yolov8n.pt")
    conf_threshold = data.get("conf_threshold", 0.3)
    target_fps = data.get("target_fps", 2)
    team_config = data.get("team_config", {
        "team1": {"color": "blue shirt", "displayColor": "#3B82F6"},
        "team2": {"color": "white shirt", "displayColor": "#FFFFFF"},
        "referee": {"color": "pink shirt", "displayColor": "#EC4899"}
    })

    # Erweiterte Parameter
    enable_team_assign = data.get("enable_team_assign", True)
    team_assign_interval = data.get("team_assign_interval", 10)  # Sekunden
    enable_jersey_ocr = data.get("enable_jersey_ocr", False)
    jersey_ocr_interval = data.get("jersey_ocr_interval", 5)  # Sekunden

    try:
        # YOLO Modell laden
        model = YOLO(model_path)

        # CLIP Modell für Team-Zuweisung laden
        clip_model, clip_processor = load_clip_model()

        # OCR Reader für Rückennummern laden
        try:
            ocr_reader = load_ocr_reader()
        except Exception as e:
            print(f"Warning: Could not load OCR reader: {e}")
            ocr_reader = None

        # Video öffnen
        cap = cv2.VideoCapture(str(clip_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame-Skip berechnen (z.B. Video 30fps, target 10fps -> jeden 3. Frame)
        frame_skip = max(1, int(video_fps / target_fps))
        actual_fps = video_fps / frame_skip

        # Track-ID -> Team Zuweisungen (akkumuliert über Zeit)
        track_team_votes = {}  # {track_id: {"team1": count, "team2": count, "referee": count}}

        # Track-ID -> Rückennummer Votes (akkumuliert über Zeit)
        track_number_votes = {}  # {track_id: {number: [(conf, frame_idx), ...]}}

        # Tracking-Daten sammeln
        tracking_results = {
            "snippet_id": snippet_id,
            "video_id": video_id,
            "video_fps": video_fps,
            "tracking_fps": actual_fps,
            "frame_skip": frame_skip,
            "total_frames": total_frames,
            "tracked_frames": 0,
            "model": model_path,
            "conf_threshold": conf_threshold,
            "field_polygon": field_polygon,
            "team_config": team_config,
            "frames": []
        }

        frame_idx = 0
        tracked_count = 0

        # Progress-Datei für Polling
        progress_file = PROJECT_ROOT / "data" / "analysis" / video_id / f"progress_{snippet_id}.json"
        def update_progress(stage, percent, detail=""):
            with open(progress_file, "w") as pf:
                json.dump({"stage": stage, "percent": percent, "detail": detail}, pf)

        update_progress("init", 0, "Starte Tracking...")
        filtered_count = 0
        frames_to_track = total_frames // frame_skip

        # Zeit-basiertes Tracking für Team-Assign und OCR
        # track_id -> letzte Timestamp für Team-Assignment
        track_last_team_assign = {}
        # track_id -> letzte Timestamp für Jersey OCR
        track_last_jersey_ocr = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Nur jeden n-ten Frame tracken
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            current_timestamp = frame_idx / video_fps

            # Progress Update
            progress_pct = int((tracked_count / max(1, frames_to_track)) * 100)
            update_progress("tracking", progress_pct, f"Frame {tracked_count}/{frames_to_track}")

            # YOLO Tracking
            results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)

            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": current_timestamp,
                "detections": []
            }

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes[i]
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                        # Prüfe ob der untere Mittelpunkt der Box im Spielfeld liegt
                        foot_x = (bbox[0] + bbox[2]) / 2  # Mitte X
                        foot_y = bbox[3]  # Unterster Punkt Y

                        in_field = True
                        if field_polygon:
                            in_field = point_in_polygon(foot_x, foot_y, field_polygon)
                            if not in_field:
                                filtered_count += 1

                        detection = {
                            "bbox": bbox,
                            "conf": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "class_name": model.names[int(box.cls[0])],
                            "in_field": in_field
                        }

                        # Track-ID falls vorhanden
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0])
                            detection["track_id"] = track_id

                        # Team-Zuweisung via CLIP (nur für Spieler im Feld)
                        if in_field and track_id is not None:
                            # Prüfe ob Team-Assignment nötig (zeitbasiert)
                            should_assign_team = enable_team_assign and clip_model is not None and clip_processor is not None
                            if should_assign_team:
                                last_assign = track_last_team_assign.get(track_id, -float('inf'))
                                if (current_timestamp - last_assign) >= team_assign_interval:
                                    try:
                                        team, team_conf = get_team_assignment(
                                            frame, bbox, team_config, clip_model, clip_processor
                                        )
                                        detection["team"] = team
                                        detection["team_conf"] = team_conf
                                        track_last_team_assign[track_id] = current_timestamp

                                        # Votes akkumulieren
                                        if track_id not in track_team_votes:
                                            track_team_votes[track_id] = {"team1": 0, "team2": 0, "referee": 0}
                                        if team in track_team_votes[track_id]:
                                            track_team_votes[track_id][team] += 1
                                    except Exception as e:
                                        print(f"Team assignment error: {e}")

                            # Rückennummer erkennen (zeitbasiert)
                            should_ocr = enable_jersey_ocr and ocr_reader is not None
                            if should_ocr:
                                last_ocr = track_last_jersey_ocr.get(track_id, -float('inf'))
                                if (current_timestamp - last_ocr) >= jersey_ocr_interval:
                                    try:
                                        jersey_num, num_conf = detect_jersey_number(frame, bbox, ocr_reader)
                                        track_last_jersey_ocr[track_id] = current_timestamp
                                        if jersey_num is not None and num_conf > 0.3:
                                            detection["jersey_number"] = jersey_num
                                            detection["jersey_conf"] = num_conf

                                            # Number votes akkumulieren
                                            if track_id not in track_number_votes:
                                                track_number_votes[track_id] = {}
                                            if jersey_num not in track_number_votes[track_id]:
                                                track_number_votes[track_id][jersey_num] = []
                                            track_number_votes[track_id][jersey_num].append({
                                                "conf": num_conf,
                                                "frame_idx": frame_idx
                                            })
                                    except Exception as e:
                                        print(f"Jersey number detection error: {e}")

                        frame_data["detections"].append(detection)

            tracking_results["frames"].append(frame_data)
            tracked_count += 1
            frame_idx += 1

        # Restliche Frames überspringen
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_idx += 1

        tracking_results["tracked_frames"] = tracked_count

        # Finale Team-Zuweisungen basierend auf Mehrheit der Votes
        final_team_assignments = {}
        for track_id, votes in track_team_votes.items():
            best_team = max(votes.keys(), key=lambda t: votes[t])
            total_votes = sum(votes.values())
            confidence = votes[best_team] / total_votes if total_votes > 0 else 0
            final_team_assignments[track_id] = {
                "team": best_team,
                "confidence": confidence,
                "votes": votes
            }

        tracking_results["track_team_assignments"] = final_team_assignments

        # Finale Rückennummer-Zuweisungen
        final_jersey_numbers = {}
        for track_id, number_votes in track_number_votes.items():
            if not number_votes:
                continue

            # Berechne gewichtete Scores für jede Nummer
            number_scores = {}
            for num, sightings in number_votes.items():
                # Gewichteter Score: Summe der Confidences
                weighted_score = sum(s["conf"] for s in sightings)
                avg_conf = weighted_score / len(sightings) if sightings else 0
                number_scores[num] = {
                    "score": weighted_score,
                    "count": len(sightings),
                    "avg_conf": avg_conf
                }

            # Sortiere nach Score (höchste zuerst)
            sorted_numbers = sorted(
                number_scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )

            # Top-Kandidaten (maximal 3)
            candidates = []
            for num, data in sorted_numbers[:3]:
                candidates.append({
                    "number": num,
                    "count": data["count"],
                    "avg_confidence": round(data["avg_conf"], 3)
                })

            if candidates:
                final_jersey_numbers[track_id] = {
                    "primary": candidates[0]["number"],
                    "primary_confidence": candidates[0]["avg_confidence"],
                    "candidates": candidates
                }

        tracking_results["track_jersey_numbers"] = final_jersey_numbers

        cap.release()

        # Tracking-Daten speichern
        tracking_dir = PROJECT_ROOT / "data" / "analysis" / video_id / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_file = tracking_dir / f"{snippet_id}.json"

        with open(tracking_file, "w") as f:
            json.dump(tracking_results, f)

        # Snippet-Status aktualisieren
        snippet["tracked"] = True
        snippet["tracking_path"] = str(tracking_file.relative_to(PROJECT_ROOT))

        with open(snippets_file, "w") as f:
            json.dump(snippets, f, indent=2)

        total_detections = sum(len(f["detections"]) for f in tracking_results["frames"])
        in_field_detections = sum(
            len([d for d in f["detections"] if d.get("in_field", True)])
            for f in tracking_results["frames"]
        )

        return jsonify({
            "success": True,
            "total_frames": total_frames,
            "tracked_frames": tracked_count,
            "tracking_fps": actual_fps,
            "detections_count": total_detections,
            "in_field_count": in_field_detections,
            "filtered_count": filtered_count,
            "has_field_polygon": field_polygon is not None
        })

    except Exception as e:
        import traceback
        print(f"Tracking error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/tracking", methods=["GET"])
def api_get_tracking_data(video_id, snippet_id):
    """Lädt Tracking-Daten für ein Snippet."""
    tracking_file = PROJECT_ROOT / "data" / "analysis" / video_id / "tracking" / f"{snippet_id}.json"

    if not tracking_file.exists():
        return jsonify({"error": "Tracking-Daten nicht gefunden"}), 404

    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)

    return jsonify(tracking_data)


@analysis_bp.route("/api/models", methods=["GET"])
def api_get_models():
    """Listet verfügbare YOLO-Modelle auf."""
    from src.db.models import TrainingRun, ActiveModel

    models = []

    # 1. Trainierte Modelle aus der Datenbank
    training_runs = db.session.query(TrainingRun).filter(
        TrainingRun.status == "completed",
        TrainingRun.model_path.isnot(None)
    ).order_by(TrainingRun.completed_at.desc()).all()

    for run in training_runs:
        model_path = PROJECT_ROOT / run.model_path if run.model_path else None
        if model_path and model_path.exists():
            models.append({
                "id": run.model_path,
                "name": run.model_name or f"Training {run.id}",
                "description": f"Trainiert am {run.completed_at.strftime('%d.%m.%Y') if run.completed_at else 'unbekannt'}",
                "type": "trained",
                "map50": run.final_map50
            })

    # 2. Lokale trainierte Modelle (models/trained/ und runs/detect/)
    trained_dirs = [
        PROJECT_ROOT / "models" / "trained",
        PROJECT_ROOT / "runs" / "detect"
    ]

    for trained_dir in trained_dirs:
        if trained_dir.exists():
            for subdir in trained_dir.iterdir():
                if subdir.is_dir():
                    best_pt = subdir / "weights" / "best.pt"
                    if best_pt.exists():
                        rel_path = str(best_pt.relative_to(PROJECT_ROOT))
                        # Überspringe wenn bereits hinzugefügt
                        if any(m["id"] == rel_path for m in models):
                            continue

                        # Versuche args.yaml zu lesen für Details
                        args_file = subdir / "args.yaml"
                        description = "Trainiertes Modell"
                        if args_file.exists():
                            try:
                                import yaml
                                with open(args_file, "r") as f:
                                    args = yaml.safe_load(f)
                                base_model = args.get("model", "?")
                                epochs = args.get("epochs", "?")
                                imgsz = args.get("imgsz", "?")
                                description = f"Basis: {base_model}, {epochs} Epochs, {imgsz}px"
                            except:
                                pass

                        # Dateigröße
                        size_mb = best_pt.stat().st_size / (1024 * 1024)

                        models.append({
                            "id": rel_path,
                            "name": subdir.name,
                            "description": f"{description} ({size_mb:.0f}MB)",
                            "type": "trained"
                        })

    # 3. Modelle im models/ Ordner (einzelne .pt Dateien)
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            rel_path = str(model_file.relative_to(PROJECT_ROOT))
            if any(m["id"] == rel_path for m in models):
                continue

            models.append({
                "id": rel_path,
                "name": model_file.stem,
                "description": "Lokales Modell",
                "type": "local"
            })

    # 4. Standard vortrainierte Modelle (immer verfügbar)
    pretrained = [
        {"id": "yolov8n.pt", "name": "YOLOv8 Nano", "description": "Schnellstes Modell, geringste Genauigkeit", "type": "pretrained"},
        {"id": "yolov8s.pt", "name": "YOLOv8 Small", "description": "Gute Balance zwischen Geschwindigkeit und Genauigkeit", "type": "pretrained"},
        {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "description": "Höhere Genauigkeit, langsamer", "type": "pretrained"},
        {"id": "yolov8l.pt", "name": "YOLOv8 Large", "description": "Hohe Genauigkeit, langsam", "type": "pretrained"},
        {"id": "yolov8x.pt", "name": "YOLOv8 Extra-Large", "description": "Beste Genauigkeit, sehr langsam", "type": "pretrained"},
    ]

    models.extend(pretrained)

    return jsonify({"models": models})


@analysis_bp.route("/clips/<video_id>/<filename>")
def serve_clip(video_id, filename):
    """Serviert extrahierte Clips."""
    clip_path = PROJECT_ROOT / "data" / "analysis" / video_id / "clips" / filename
    if not clip_path.exists():
        abort(404)
    return send_file(clip_path, mimetype="video/mp4")


@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/positions", methods=["GET"])
def api_get_positions(video_id, snippet_id):
    """
    Berechnet die Spielfeld-Positionen für alle Tracking-Daten.
    Nutzt die Kalibrierung (Homography + Fisheye-Korrektur).

    Query Parameter:
        calibration_id: (optional) ID einer spezifischen Kalibrierung
    """
    import cv2
    import numpy as np
    from src.processing.calibration import transform_point_with_distortion

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Kalibrierungs-ID aus Query-Parameter (falls angegeben)
    calibration_id = request.args.get("calibration_id")

    if calibration_id:
        # Spezifische Kalibrierung laden
        calibration = db.session.get(Calibration, calibration_id)
        if not calibration:
            return jsonify({"error": f"Kalibrierung {calibration_id} nicht gefunden"}), 404
    else:
        # Fallback: Aktive Kalibrierung für dieses Video
        calibration = db.session.query(Calibration).filter_by(
            video_id=video.id,
            is_active=True
        ).first()

        # Falls keine aktive Kalibrierung für dieses Video, nimm die neueste aktive
        if not calibration:
            calibration = db.session.query(Calibration).filter_by(
                is_active=True
            ).order_by(Calibration.created_at.desc()).first()

    if not calibration or not calibration.calibration_data:
        return jsonify({"error": "Keine Kalibrierung gefunden. Bitte zuerst eine Kalibrierung erstellen."}), 404

    cal_data = calibration.calibration_data
    print(f"[Positions] Using calibration: {calibration.name} (id={calibration.id})")

    # Debug: Kalibrierungsdaten anzeigen
    print(f"[Positions] cal_data keys: {cal_data.keys()}")

    # Homography-Matrix extrahieren
    homography_matrix = None
    if cal_data.get("homography_matrix"):
        homography_matrix = np.array(cal_data.get("homography_matrix"), dtype=np.float32)
        print(f"[Positions] Using direct homography_matrix")
    elif cal_data.get("combined_calibration", {}).get("homography_matrix"):
        homography_matrix = np.array(cal_data["combined_calibration"]["homography_matrix"], dtype=np.float32)
        print(f"[Positions] Using combined_calibration homography_matrix")
    else:
        return jsonify({"error": "Keine Homography-Matrix in Kalibrierung gefunden"}), 404

    print(f"[Positions] Homography matrix shape: {homography_matrix.shape}")

    # Undistort-Parameter extrahieren (für Fisheye-Entzerrung)
    undistort_params = cal_data.get("undistort_params", {})
    k1 = undistort_params.get("k1", 0.0)
    k2 = undistort_params.get("k2", 0.0)
    k3 = undistort_params.get("k3", 0.0)
    k4 = undistort_params.get("k4", 0.0)
    zoom_out = undistort_params.get("zoom_out", 1.0)
    rotation = undistort_params.get("rotation", 0.0)
    lens_profile_id = undistort_params.get("lens_profile_id") or cal_data.get("lens_profile_id")

    print(f"[Positions] Undistort params: k1={k1}, k2={k2}, k3={k3}, k4={k4}, zoom_out={zoom_out}, rotation={rotation}")
    print(f"[Positions] Lens profile: {lens_profile_id}")

    # Lens-Profil laden für Kamera-Matrix (falls vorhanden)
    from src.processing.calibration import load_lens_profile, scale_camera_matrix
    camera_matrix = None
    dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float64)
    profile_resolution = None

    # Bei Custom-Modus: base_profile_id verwenden
    base_profile_id = undistort_params.get("base_profile_id")
    profile_to_load = base_profile_id if lens_profile_id == "custom" else lens_profile_id

    if profile_to_load:
        profile = load_lens_profile(profile_to_load)
        if profile and profile.get("fisheye_params"):
            camera_matrix = np.array(profile["fisheye_params"]["camera_matrix"], dtype=np.float64)
            profile_resolution = profile.get("resolution", {})
            # Wenn k1-k4 nicht in undistort_params und nicht custom, aus Profil nehmen
            if lens_profile_id != "custom" and k1 == 0 and k2 == 0:
                coeffs = profile["fisheye_params"].get("distortion_coeffs", [0, 0, 0, 0])
                dist_coeffs = np.array(coeffs[:4], dtype=np.float64)
            print(f"[Positions] Loaded camera_matrix from profile: {profile_to_load}")
            print(f"[Positions] Profile resolution: {profile_resolution}")

    # Video-Auflösung ermitteln (aus Video-Metadaten oder Standard)
    video_width = video.width if hasattr(video, 'width') and video.width else 1920
    video_height = video.height if hasattr(video, 'height') and video.height else 1080
    print(f"[Positions] Video resolution: {video_width}x{video_height}")

    # Kamera-Matrix auf Video-Auflösung skalieren falls nötig
    if camera_matrix is not None and profile_resolution:
        profile_w = profile_resolution.get("width", video_width)
        profile_h = profile_resolution.get("height", video_height)
        if profile_w != video_width or profile_h != video_height:
            print(f"[Positions] Scaling camera_matrix from {profile_w}x{profile_h} to {video_width}x{video_height}")
            camera_matrix = scale_camera_matrix(camera_matrix, (profile_w, profile_h), (video_width, video_height))
            print(f"[Positions] Scaled camera_matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}, cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")

    # Fallback: Standard-Kamera-Matrix basierend auf Video-Grösse
    if camera_matrix is None:
        focal = video_width  # Typische Fisheye-Brennweite
        cx, cy = video_width / 2, video_height / 2
        camera_matrix = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        print(f"[Positions] Using default camera_matrix: focal={focal}, center=({cx}, {cy})")

    # Tracking-Daten laden
    tracking_file = PROJECT_ROOT / "data" / "analysis" / video_id / "tracking" / f"{snippet_id}.json"
    if not tracking_file.exists():
        return jsonify({"error": "Tracking-Daten nicht gefunden"}), 404

    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)

    # Team-Assignments laden
    track_team_assignments = tracking_data.get("track_team_assignments", {})

    # Floorball-Spielfeld Dimensionen (für Validierung)
    FIELD_LENGTH = 40.0
    FIELD_WIDTH = 20.0

    # Positionen für jeden Frame berechnen
    positions_data = {
        "snippet_id": snippet_id,
        "video_id": video_id,
        "field_dimensions": {"length": FIELD_LENGTH, "width": FIELD_WIDTH},
        "calibration_used": {
            "id": str(calibration.id),
            "name": calibration.name,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "zoom_out": zoom_out,
            "rotation": rotation,
            "lens_profile_id": lens_profile_id
        },
        "frames": []
    }

    # Import für Punkt-Entzerrung
    from src.processing.calibration import undistort_points

    def pixel_to_field(px, py):
        """
        Transformiert verzerrte Pixel-Koordinaten zu Feld-Koordinaten.

        Der Prozess bei der Bildentzerrung war:
        1. Fisheye-Entzerrung mit new_K (Brennweite / zoom_out)
        2. Rotation anwenden

        Bei der Punkt-Transformation müssen wir das gleiche machen:
        1. Fisheye-Entzerrung mit angepasster Kamera-Matrix (zoom_out eingebaut)
        2. Rotation anwenden
        3. Homography anwenden
        """
        w, h = video_width, video_height
        cx_img, cy_img = w / 2, h / 2

        undist_px, undist_py = px, py

        # 1. Fisheye-Entzerrung mit zoom_out in der Kamera-Matrix
        # Bei der Bildentzerrung wurde: new_K[0,0] = K[0,0] / zoom_out
        # undistortPoints verwendet aber P (=new_K), also müssen wir das gleiche tun
        if camera_matrix is not None and (k1 != 0 or k2 != 0 or k3 != 0 or k4 != 0):
            dist_coeffs_arr = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(4, 1)

            # Angepasste Kamera-Matrix für Output (mit zoom_out)
            new_K = camera_matrix.copy()
            if zoom_out != 1.0:
                new_K[0, 0] /= zoom_out
                new_K[1, 1] /= zoom_out

            pts = np.array([[[px, py]]], dtype=np.float64)

            # undistortPoints: Input K, dist_coeffs, R=eye, P=new_K
            undistorted = cv2.fisheye.undistortPoints(
                pts,
                camera_matrix,
                dist_coeffs_arr,
                R=np.eye(3),
                P=new_K
            )
            undist_px, undist_py = undistorted[0, 0]

        # 2. Rotation rückgängig machen (NEGATIVER Winkel!)
        # Das entzerrte Bild wurde um +rotation Grad rotiert,
        # also müssen wir die Punkte um -rotation Grad rotieren
        if abs(rotation) > 0.01:
            angle_rad = np.radians(-rotation)  # NEGATIV!
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            centered_x = undist_px - cx_img
            centered_y = undist_py - cy_img
            rot_x = centered_x * cos_a - centered_y * sin_a
            rot_y = centered_x * sin_a + centered_y * cos_a
            undist_px = rot_x + cx_img
            undist_py = rot_y + cy_img

        # 3. Homography anwenden (entzerrte Pixel → Feld)
        pts = np.array([[[undist_px, undist_py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, homography_matrix)
        field_x, field_y = transformed[0, 0]

        return float(field_x), float(field_y)

    # Debug: Sammle Statistiken
    total_detections = 0
    skipped_not_in_field = 0
    skipped_transform_error = 0
    skipped_out_of_bounds = 0
    valid_positions = 0

    for frame in tracking_data.get("frames", []):
        frame_positions = {
            "frame_idx": frame["frame_idx"],
            "timestamp": frame["timestamp"],
            "players": [],
            "referees": [],
            "goalkeepers": [],
            "unknown": []
        }

        for det in frame.get("detections", []):
            total_detections += 1

            if not det.get("in_field", True):
                skipped_not_in_field += 1
                continue

            # Fuss-Position (untere Mitte der Bounding Box)
            bbox = det["bbox"]
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]

            # Zu Feld-Koordinaten transformieren
            try:
                field_x, field_y = pixel_to_field(foot_x, foot_y)
            except Exception as e:
                skipped_transform_error += 1
                print(f"[Positions] Transform error for pixel ({foot_x:.0f}, {foot_y:.0f}): {e}")
                continue

            # Debug: Erste paar Transformationen loggen
            if total_detections <= 5:
                print(f"[Positions] Pixel ({foot_x:.0f}, {foot_y:.0f}) -> Field ({field_x:.2f}, {field_y:.2f})")

            # Validierung: Punkt sollte auf dem Spielfeld sein (mit grosser Toleranz für Debug)
            # Normale Toleranz wäre -2 bis +2, aber wir loggen auch was ausserhalb ist
            if field_x < -5 or field_x > FIELD_LENGTH + 5 or field_y < -5 or field_y > FIELD_WIDTH + 5:
                skipped_out_of_bounds += 1
                if skipped_out_of_bounds <= 10:
                    print(f"[Positions] Out of bounds: ({field_x:.2f}, {field_y:.2f}) from pixel ({foot_x:.0f}, {foot_y:.0f})")
                continue

            valid_positions += 1

            track_id = det.get("track_id")

            # Team aus finalen Assignments holen
            team = "unknown"
            if track_id is not None:
                track_id_str = str(track_id)
                if track_id_str in track_team_assignments:
                    team = track_team_assignments[track_id_str].get("team", "unknown")
                elif det.get("team"):
                    team = det["team"]

            position = {
                "track_id": track_id,
                "x": round(field_x, 2),
                "y": round(field_y, 2),
                "team": team,
                "confidence": det.get("conf", 0),
                "jersey_number": det.get("jersey_number")
            }

            # Nach Typ kategorisieren
            class_name = det.get("class_name", "").lower()
            if "goalkeeper" in class_name or "goalie" in class_name:
                frame_positions["goalkeepers"].append(position)
            elif team == "referee":
                frame_positions["referees"].append(position)
            elif team in ["team1", "team2"]:
                frame_positions["players"].append(position)
            else:
                frame_positions["unknown"].append(position)

        positions_data["frames"].append(frame_positions)

    # Debug-Statistiken
    print(f"[Positions] Stats: total={total_detections}, not_in_field={skipped_not_in_field}, "
          f"transform_error={skipped_transform_error}, out_of_bounds={skipped_out_of_bounds}, valid={valid_positions}")

    # Stats zum Response hinzufügen
    positions_data["debug_stats"] = {
        "total_detections": total_detections,
        "skipped_not_in_field": skipped_not_in_field,
        "skipped_transform_error": skipped_transform_error,
        "skipped_out_of_bounds": skipped_out_of_bounds,
        "valid_positions": valid_positions
    }

    # Inverse Homography für Spielfeld-Grid-Overlay berechnen
    try:
        inv_homography = np.linalg.inv(homography_matrix)
        positions_data["inverse_homography"] = inv_homography.tolist()

        # Vollständige Entzerrungs-Parameter für Grid-Overlay
        positions_data["distortion"] = {
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "zoom_out": zoom_out,
            "rotation": rotation,
            "camera_matrix": camera_matrix.tolist() if camera_matrix is not None else None,
            "lens_profile_id": lens_profile_id
        }
        print(f"[Positions] Distortion params for grid: k1={k1}, k2={k2}, k3={k3}, k4={k4}, zoom={zoom_out}, rot={rotation}")

        # Grid-Linien im Backend berechnen (Spielfeld → verzerrte Pixel)
        # Verwendet die GLEICHE Logik wie api_distorted_overlay in calibration.py

        # Die gezoomte Kameramatrix (wie bei der Entzerrung verwendet)
        zoomed_K = camera_matrix.copy()
        zoomed_K[0, 0] /= zoom_out
        zoomed_K[1, 1] /= zoom_out

        # WICHTIG: distortion_coeffs als (4,1) Array wie in calibration.py!
        dist_coeffs_arr = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(4, 1)

        def transform_and_distort_line(field_points):
            """
            Transformiert Feldpunkte zu Bildpunkten mit Fisheye-Verzerrung.

            Workflow (GLEICH wie in calibration.py api_distorted_overlay):
            1. Feldpunkte → entzerrte Bildpunkte (via inverse Homographie)
            2. Rotation rückgängig machen (ZUERST, mit POSITIVEM Winkel!)
            3. Von gezoomter Kameramatrix zu normalisierten Koordinaten
            4. Fisheye-Verzerrung anwenden via projectPoints
            """
            w, h = video_width, video_height
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            # 1. Feld → Bild (entzerrt + rotiert, mit Zoom)
            field_pts_arr = np.array([[[p[0], p[1]]] for p in field_points], dtype=np.float32)
            undistorted_pts = cv2.perspectiveTransform(field_pts_arr, inv_homography)
            undistorted_pts_2d = undistorted_pts.reshape(-1, 2)

            # 2. Rotation rückgängig machen (ZUERST, bevor Fisheye!)
            # WICHTIG: POSITIVER Winkel, nicht negativ!
            if abs(rotation) > 0.01:
                center = np.array([w / 2, h / 2])
                angle_rad = np.radians(rotation)  # POSITIV!
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rotated_pts = np.zeros_like(undistorted_pts_2d)
                for i, pt in enumerate(undistorted_pts_2d):
                    translated = pt - center
                    rotated_pts[i, 0] = translated[0] * cos_a - translated[1] * sin_a + center[0]
                    rotated_pts[i, 1] = translated[0] * sin_a + translated[1] * cos_a + center[1]
                undistorted_pts_2d = rotated_pts

            # 3. Von gezoomter Kameramatrix zu normalisierten Koordinaten
            fx_zoom, fy_zoom = zoomed_K[0, 0], zoomed_K[1, 1]
            normalized = np.zeros_like(undistorted_pts_2d)
            normalized[:, 0] = (undistorted_pts_2d[:, 0] - cx) / fx_zoom
            normalized[:, 1] = (undistorted_pts_2d[:, 1] - cy) / fy_zoom

            # 4. Fisheye-Verzerrung anwenden
            pts_3d = np.zeros((len(normalized), 3), dtype=np.float32)
            pts_3d[:, 0] = normalized[:, 0]
            pts_3d[:, 1] = normalized[:, 1]
            pts_3d[:, 2] = 1.0

            rvec = np.zeros(3, dtype=np.float32)
            tvec = np.zeros(3, dtype=np.float32)

            distorted, _ = cv2.fisheye.projectPoints(
                pts_3d.reshape(-1, 1, 3),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs_arr
            )

            return distorted.reshape(-1, 2)

        def generate_line_points(start, end, segment_length=0.5):
            """Generiert Punkte entlang einer Linie mit gegebenem Segment-Abstand."""
            dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            num_segments = max(int(dist / segment_length), 2)
            points = []
            for i in range(num_segments + 1):
                t = i / num_segments
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                points.append([x, y])
            return points

        def field_to_distorted_pixel(field_points):
            """Wrapper für Kompatibilität."""
            return transform_and_distort_line(field_points)

        # Spielfeld-Linien definieren (in Metern)
        # WICHTIG: Linien müssen in kleine Segmente aufgeteilt werden für korrekte Fisheye-Krümmung!
        L, W = 40.0, 20.0
        SEGMENT_LENGTH = 0.5  # 0.5m pro Segment für glatte Kurven
        grid_lines = []

        # Spielfeld-Rand (4 Seiten, jede als eigene segmentierte Linie)
        # Untere Seite: (0,0) → (L,0)
        bottom_pts = generate_line_points([0, 0], [L, 0], SEGMENT_LENGTH)
        bottom_pixels = field_to_distorted_pixel(bottom_pts)
        grid_lines.append({"type": "boundary", "points": bottom_pixels.tolist()})

        # Rechte Seite: (L,0) → (L,W)
        right_pts = generate_line_points([L, 0], [L, W], SEGMENT_LENGTH)
        right_pixels = field_to_distorted_pixel(right_pts)
        grid_lines.append({"type": "boundary", "points": right_pixels.tolist()})

        # Obere Seite: (L,W) → (0,W)
        top_pts = generate_line_points([L, W], [0, W], SEGMENT_LENGTH)
        top_pixels = field_to_distorted_pixel(top_pts)
        grid_lines.append({"type": "boundary", "points": top_pixels.tolist()})

        # Linke Seite: (0,W) → (0,0)
        left_pts = generate_line_points([0, W], [0, 0], SEGMENT_LENGTH)
        left_pixels = field_to_distorted_pixel(left_pts)
        grid_lines.append({"type": "boundary", "points": left_pixels.tolist()})

        # Mittellinie (segmentiert)
        midline_pts = generate_line_points([L/2, 0], [L/2, W], SEGMENT_LENGTH)
        midline_pixels = field_to_distorted_pixel(midline_pts)
        grid_lines.append({"type": "midline", "points": midline_pixels.tolist()})

        # Torlinien (2.85m von der Endlinie, segmentiert)
        goal_dist = 2.85
        left_goal_pts = generate_line_points([goal_dist, 0], [goal_dist, W], SEGMENT_LENGTH)
        left_goal_pixels = field_to_distorted_pixel(left_goal_pts)
        grid_lines.append({"type": "goal_line", "points": left_goal_pixels.tolist()})

        right_goal_pts = generate_line_points([L - goal_dist, 0], [L - goal_dist, W], SEGMENT_LENGTH)
        right_goal_pixels = field_to_distorted_pixel(right_goal_pts)
        grid_lines.append({"type": "goal_line", "points": right_goal_pixels.tolist()})

        # Mittelkreis (3m Radius, mit vielen Punkten für glatte Kurve)
        center_x, center_y = L/2, W/2
        circle_points = []
        for angle in range(0, 361, 6):  # 6° Schritte für glatte Kurve
            rad = np.radians(angle)
            circle_points.append([center_x + 3 * np.cos(rad), center_y + 3 * np.sin(rad)])
        circle_pixels = field_to_distorted_pixel(circle_points)
        grid_lines.append({"type": "center_circle", "points": circle_pixels.tolist()})

        # Mittelpunkt
        center_pixel = field_to_distorted_pixel([[center_x, center_y]])
        grid_lines.append({"type": "center_point", "points": center_pixel.tolist()})

        # Torraum-Halbkreise (4m Radius, mehr Punkte)
        goal_radius = 4.0
        # Linker Torraum
        left_crease = []
        for angle in range(-90, 91, 6):
            rad = np.radians(angle)
            left_crease.append([goal_dist + goal_radius * np.cos(rad), center_y + goal_radius * np.sin(rad)])
        left_crease_pixels = field_to_distorted_pixel(left_crease)
        grid_lines.append({"type": "crease", "points": left_crease_pixels.tolist()})

        # Rechter Torraum
        right_crease = []
        for angle in range(90, 271, 6):
            rad = np.radians(angle)
            right_crease.append([L - goal_dist + goal_radius * np.cos(rad), center_y + goal_radius * np.sin(rad)])
        right_crease_pixels = field_to_distorted_pixel(right_crease)
        grid_lines.append({"type": "crease", "points": right_crease_pixels.tolist()})

        positions_data["grid_lines"] = grid_lines
        print(f"[Positions] Generated {len(grid_lines)} grid elements for overlay")

    except Exception as e:
        import traceback
        print(f"[Positions] Could not compute inverse homography/grid: {e}")
        traceback.print_exc()

    # Positions-Daten cachen
    positions_file = PROJECT_ROOT / "data" / "analysis" / video_id / "positions" / f"{snippet_id}.json"
    positions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(positions_file, "w") as f:
        json.dump(positions_data, f)

    return jsonify(positions_data)


@analysis_bp.route("/video/<video_id>/tactical/<snippet_id>")
def tactical_view(video_id, snippet_id):
    """Taktische Ansicht mit synchronisiertem Video und Spielfeld."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    # Snippet-Daten laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        abort(404)

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        abort(404)

    # Team-Config laden (aus Kalibrierung oder Default)
    team_config = {
        "team1": {"displayColor": "#3B82F6"},
        "team2": {"displayColor": "#FFFFFF"},
        "referee": {"displayColor": "#EC4899"}
    }

    return render_template(
        "analysis/tactical.html",
        video=video,
        snippet=snippet,
        team_config=team_config
    )
