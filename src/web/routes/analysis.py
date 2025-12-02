"""
Analysis Routes - Video-Analyse mit Tracking

Refaktoriert: Verwendet Module aus src/trackers und src/analysis
"""
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, abort, send_file
from src.web.extensions import db
from src.db.models import Video, Calibration, GamePeriod, AnalysisChunk

# Modularisierte Komponenten
from src.trackers import YOLOTracker, point_in_polygon
from src.analysis import TeamAssigner, JerseyOCR, PositionMapper, GridGenerator

analysis_bp = Blueprint("analysis", __name__, url_prefix="/analysis")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# === View Routes ===

@analysis_bp.route("/")
def index():
    """Analyse-Übersicht - Videos mit Kalibrierung."""
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

    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    all_calibrations = db.session.query(Calibration).filter_by(
        video_id=video.id
    ).order_by(Calibration.created_at.desc()).all()

    # Bestehende Snippets laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    snippets = []
    if snippets_file.exists():
        with open(snippets_file, "r") as f:
            snippets = json.load(f)

    # Tracker-Konfigurationen laden
    trackers = {}
    yolo_config_file = PROJECT_ROOT / "configs" / "yolo_models.yaml"
    if yolo_config_file.exists():
        import yaml
        with open(yolo_config_file, "r") as f:
            yolo_config = yaml.safe_load(f)
            trackers = yolo_config.get("trackers", {})

    return render_template(
        "analysis/video.html",
        video=video,
        calibration=calibration,
        all_calibrations=all_calibrations,
        snippets=snippets,
        trackers=trackers
    )


@analysis_bp.route("/video/<video_id>/review/<snippet_id>")
def review_tracking(video_id, snippet_id):
    """Tracking-Review-Seite für ein Snippet."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        abort(404)

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        abort(404)

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


@analysis_bp.route("/video/<video_id>/tactical/<snippet_id>")
def tactical_view(video_id, snippet_id):
    """Taktische Ansicht mit synchronisiertem Video und Spielfeld."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        abort(404)

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        abort(404)

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


@analysis_bp.route("/video/<video_id>/tactical-full")
def tactical_view_full(video_id):
    """Taktische Ansicht für die volle Spielanalyse (alle Chunks)."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    # Kalibrierung laden
    calibration_id = request.args.get("calibration_id")
    calibration = None
    if calibration_id:
        calibration = db.session.get(Calibration, calibration_id)
    else:
        calibration = db.session.query(Calibration).filter_by(
            video_id=video.id,
            is_active=True
        ).first()

    # Spielzeiten laden
    periods = db.session.query(GamePeriod).filter_by(
        video_id=video.id
    ).order_by(GamePeriod.period_index).all()

    # Chunks laden
    chunks = db.session.query(AnalysisChunk).filter_by(
        video_id=video.id,
        status="completed"
    ).order_by(AnalysisChunk.chunk_index).all()

    # Gesamtdauer berechnen
    total_duration = sum(p.end_time - p.start_time for p in periods)

    # Team-Konfiguration laden (aus gespeicherten Snippets)
    team_config = {
        "team1": {"displayColor": "#3B82F6"},
        "team2": {"displayColor": "#FFFFFF"},
        "referee": {"displayColor": "#EC4899"}
    }

    # Versuche team_config aus Snippets zu laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if snippets_file.exists():
        try:
            with open(snippets_file, "r") as f:
                snippets = json.load(f)
                if snippets and isinstance(snippets, list) and len(snippets) > 0:
                    first_snippet = snippets[0]
                    if "team_config" in first_snippet:
                        team_config = first_snippet["team_config"]
        except Exception as e:
            print(f"Could not load team config: {e}")

    return render_template(
        "analysis/tactical_full.html",
        video=video,
        calibration=calibration,
        periods=periods,
        chunks=chunks,
        total_duration=total_duration,
        team_config=team_config
    )


# === API Endpoints - Snippets ===

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

    if len(snippets) > 5:
        return jsonify({"error": "Maximal 5 Snippets erlaubt"}), 400

    for snippet in snippets:
        duration = snippet.get("end", 0) - snippet.get("start", 0)
        if duration < 5 or duration > 30:
            return jsonify({"error": f"Snippet-Dauer muss zwischen 5 und 30 Sekunden liegen (aktuell: {duration:.1f}s)"}), 400

    analysis_dir = PROJECT_ROOT / "data" / "analysis" / video_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for i, snippet in enumerate(snippets):
        if not snippet.get("id"):
            snippet["id"] = f"snippet_{i+1}_{int(datetime.now().timestamp())}"

    snippets_file = analysis_dir / "snippets.json"
    with open(snippets_file, "w") as f:
        json.dump(snippets, f, indent=2)

    return jsonify({"success": True, "snippets": snippets})


# === API Endpoints - Boundary ===

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

    if len(boundary_polygon) < 3:
        return jsonify({"error": "Mindestens 3 Punkte erforderlich"}), 400

    analysis_dir = PROJECT_ROOT / "data" / "analysis" / video_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    boundary_file = analysis_dir / "boundary.json"
    with open(boundary_file, "w") as f:
        json.dump({"boundary_polygon": boundary_polygon}, f, indent=2)

    return jsonify({"success": True, "boundary_polygon": boundary_polygon})


# === API Endpoints - Progress & Extraction ===

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

    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        return jsonify({"error": "Keine Snippets gefunden"}), 404

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        return jsonify({"error": "Snippet nicht gefunden"}), 404

    video_path = PROJECT_ROOT / video.file_path
    output_dir = PROJECT_ROOT / "data" / "analysis" / video_id / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{snippet_id}.mp4"

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


# === API Endpoints - Tracking ===

@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/track", methods=["POST"])
def api_track_snippet(video_id, snippet_id):
    """
    Führt YOLO-Tracking auf einem Snippet aus.

    Verwendet die modularisierten Komponenten:
    - YOLOTracker für Detection und Tracking
    - TeamAssigner für CLIP-basierte Team-Zuweisung
    - JerseyOCR für Rückennummer-Erkennung
    """
    import cv2

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Spielfeld-Polygon laden
    field_polygon = _load_field_polygon(video_id, video.id)

    # Snippet-Daten laden
    snippets_file = PROJECT_ROOT / "data" / "analysis" / video_id / "snippets.json"
    if not snippets_file.exists():
        return jsonify({"error": "Keine Snippets gefunden"}), 404

    with open(snippets_file, "r") as f:
        snippets = json.load(f)

    snippet = next((s for s in snippets if s["id"] == snippet_id), None)
    if not snippet:
        return jsonify({"error": "Snippet nicht gefunden"}), 404

    clip_path = PROJECT_ROOT / snippet.get("clip_path", "")
    if not clip_path.exists():
        return jsonify({"error": "Clip nicht gefunden. Bitte zuerst extrahieren."}), 404

    # Request-Parameter
    data = request.get_json() or {}
    model_path = data.get("model_path", "yolov8n.pt")
    tracker_type = data.get("tracker", "bytetrack")
    conf_threshold = data.get("conf_threshold", 0.3)
    target_fps = data.get("target_fps", 2)
    team_config = data.get("team_config", {
        "team1": {"color": "blue shirt", "displayColor": "#3B82F6"},
        "team2": {"color": "white shirt", "displayColor": "#FFFFFF"},
        "referee": {"color": "pink shirt", "displayColor": "#EC4899"}
    })

    # Tracker-Konfiguration
    tracker_config = _get_tracker_config(tracker_type)

    # Erweiterte Parameter
    enable_team_assign = data.get("enable_team_assign", True)
    team_assign_interval = data.get("team_assign_interval", 10)
    enable_jersey_ocr = data.get("enable_jersey_ocr", False)
    jersey_ocr_interval = data.get("jersey_ocr_interval", 5)

    try:
        # Progress-Tracking
        progress_file = PROJECT_ROOT / "data" / "analysis" / video_id / f"progress_{snippet_id}.json"

        def update_progress(stage, percent, detail=""):
            with open(progress_file, "w") as pf:
                json.dump({"stage": stage, "percent": percent, "detail": detail}, pf)

        update_progress("init", 0, "Initialisiere Tracker...")

        # Komponenten initialisieren
        tracker = YOLOTracker(
            model_path=model_path,
            tracker_config=tracker_config,
            conf_threshold=conf_threshold
        )

        team_assigner = None
        if enable_team_assign:
            team_assigner = TeamAssigner(team_config=team_config)
            if not team_assigner.is_available:
                print("Warning: CLIP model not available for team assignment")
                team_assigner = None

        jersey_detector = None
        if enable_jersey_ocr:
            jersey_detector = JerseyOCR()
            if not jersey_detector.is_available:
                print("Warning: OCR not available for jersey detection")
                jersey_detector = None

        update_progress("tracking", 0, "Starte Tracking...")

        # Progress-Callback
        def on_frame(current, total, detections):
            if total > 0:
                percent = int((current / total) * 100)
                update_progress("tracking", percent, f"Frame {current}/{total}")

        # Tracking durchführen
        tracking_results = tracker.track_video(
            video_path=str(clip_path),
            target_fps=target_fps,
            field_polygon=field_polygon,
            on_frame=on_frame,
            team_assigner=team_assigner,
            jersey_detector=jersey_detector,
            team_assign_interval=team_assign_interval,
            jersey_ocr_interval=jersey_ocr_interval
        )

        # Zusätzliche Metadaten
        tracking_results["snippet_id"] = snippet_id
        tracking_results["video_id"] = video_id
        tracking_results["team_config"] = team_config

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

        update_progress("done", 100, "Tracking abgeschlossen")

        # Statistiken
        total_detections = sum(len(f["detections"]) for f in tracking_results["frames"])
        in_field_detections = sum(
            len([d for d in f["detections"] if d.get("in_field", True)])
            for f in tracking_results["frames"]
        )

        return jsonify({
            "success": True,
            "total_frames": tracking_results["total_frames"],
            "tracked_frames": tracking_results["tracked_frames"],
            "tracking_fps": tracking_results["tracking_fps"],
            "detections_count": total_detections,
            "in_field_count": in_field_detections,
            "filtered_count": tracking_results.get("filtered_count", 0),
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


# === API Endpoints - Positions ===

@analysis_bp.route("/api/videos/<video_id>/snippets/<snippet_id>/positions", methods=["GET"])
def api_get_positions(video_id, snippet_id):
    """
    Berechnet die Spielfeld-Positionen für alle Tracking-Daten.

    Verwendet die modularisierten Komponenten:
    - PositionMapper für Pixel-zu-Feld Transformation
    - GridGenerator für Spielfeld-Overlay
    """
    from src.processing.calibration import load_lens_profile

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Kalibrierung laden
    calibration = _load_calibration(video_id, video.id, request.args.get("calibration_id"))
    if not calibration or not calibration.calibration_data:
        return jsonify({"error": "Keine Kalibrierung gefunden"}), 404

    cal_data = calibration.calibration_data
    print(f"[Positions] Using calibration: {calibration.name} (id={calibration.id})")

    # Video-Auflösung
    video_width = video.width if hasattr(video, 'width') and video.width else 1920
    video_height = video.height if hasattr(video, 'height') and video.height else 1080

    # Tracking-Daten laden
    tracking_file = PROJECT_ROOT / "data" / "analysis" / video_id / "tracking" / f"{snippet_id}.json"
    if not tracking_file.exists():
        return jsonify({"error": "Tracking-Daten nicht gefunden"}), 404

    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)

    try:
        # Position Mapper erstellen
        position_mapper = PositionMapper.from_calibration_data(
            cal_data=cal_data,
            video_width=video_width,
            video_height=video_height,
            lens_profile_loader=load_lens_profile
        )

        # Positionen transformieren
        positions_data = position_mapper.transform_tracking_data(tracking_data)

        # Metadaten hinzufügen
        positions_data["snippet_id"] = snippet_id
        positions_data["video_id"] = video_id
        positions_data["calibration_used"] = {
            "id": str(calibration.id),
            "name": calibration.name
        }

        # Grid Generator erstellen
        grid_generator = GridGenerator.from_calibration_data(
            cal_data=cal_data,
            video_width=video_width,
            video_height=video_height,
            lens_profile_loader=load_lens_profile
        )

        # Grid-Linien generieren
        positions_data["grid_lines"] = grid_generator.generate_grid_lines()

        # Distortion-Parameter für Frontend
        undistort_params = cal_data.get("undistort_params", {})
        positions_data["distortion"] = {
            "k1": undistort_params.get("k1", 0.0),
            "k2": undistort_params.get("k2", 0.0),
            "k3": undistort_params.get("k3", 0.0),
            "k4": undistort_params.get("k4", 0.0),
            "zoom_out": undistort_params.get("zoom_out", 1.0),
            "rotation": undistort_params.get("rotation", 0.0)
        }

        # Debug-Statistiken loggen
        stats = positions_data.get("debug_stats", {})
        print(f"[Positions] Stats: total={stats.get('total_detections', 0)}, "
              f"valid={stats.get('valid_positions', 0)}, "
              f"out_of_bounds={stats.get('skipped_out_of_bounds', 0)}")

        # Positions-Daten cachen
        positions_file = PROJECT_ROOT / "data" / "analysis" / video_id / "positions" / f"{snippet_id}.json"
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(positions_file, "w") as f:
            json.dump(positions_data, f)

        return jsonify(positions_data)

    except Exception as e:
        import traceback
        print(f"[Positions] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# === API Endpoints - Models ===

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

    # 2. Lokale trainierte Modelle
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
                        if any(m["id"] == rel_path for m in models):
                            continue

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

                        size_mb = best_pt.stat().st_size / (1024 * 1024)

                        models.append({
                            "id": rel_path,
                            "name": subdir.name,
                            "description": f"{description} ({size_mb:.0f}MB)",
                            "type": "trained"
                        })

    # 3. Modelle im models/ Ordner
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

    # 4. Standard vortrainierte Modelle
    pretrained = [
        {"id": "yolov8n.pt", "name": "YOLOv8 Nano", "description": "Schnellstes Modell, geringste Genauigkeit", "type": "pretrained"},
        {"id": "yolov8s.pt", "name": "YOLOv8 Small", "description": "Gute Balance zwischen Geschwindigkeit und Genauigkeit", "type": "pretrained"},
        {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "description": "Höhere Genauigkeit, langsamer", "type": "pretrained"},
        {"id": "yolov8l.pt", "name": "YOLOv8 Large", "description": "Hohe Genauigkeit, langsam", "type": "pretrained"},
        {"id": "yolov8x.pt", "name": "YOLOv8 Extra-Large", "description": "Beste Genauigkeit, sehr langsam", "type": "pretrained"},
    ]

    models.extend(pretrained)

    return jsonify({"models": models})


# === Static File Serving ===

@analysis_bp.route("/clips/<video_id>/<filename>")
def serve_clip(video_id, filename):
    """Serviert extrahierte Clips."""
    clip_path = PROJECT_ROOT / "data" / "analysis" / video_id / "clips" / filename
    if not clip_path.exists():
        abort(404)
    return send_file(clip_path, mimetype="video/mp4")


# === Helper Functions ===

def _load_field_polygon(video_id: str, video_db_id: str):
    """Lädt das Spielfeld-Polygon aus boundary.json oder Kalibrierung."""
    field_polygon = None

    # Primär aus boundary.json
    boundary_file = PROJECT_ROOT / "data" / "analysis" / video_id / "boundary.json"
    if boundary_file.exists():
        with open(boundary_file, "r") as f:
            boundary_data = json.load(f)
            field_polygon = boundary_data.get("boundary_polygon", [])
            if len(field_polygon) < 3:
                field_polygon = None

    # Fallback: aus Kalibrierung
    if field_polygon is None:
        calibration = db.session.query(Calibration).filter_by(
            video_id=video_db_id,
            is_active=True
        ).first()
        if calibration and calibration.calibration_data:
            step2 = calibration.calibration_data.get("step2_boundary", {})
            field_polygon = step2.get("boundary_polygon", [])
            if len(field_polygon) < 3:
                field_polygon = None

    return field_polygon


def _get_tracker_config(tracker_type: str):
    """Gibt den Pfad zur Tracker-Konfiguration zurück."""
    tracker_configs = {
        "bytetrack": PROJECT_ROOT / "configs" / "trackers" / "bytetrack.yaml",
        "botsort": PROJECT_ROOT / "configs" / "trackers" / "botsort.yaml",
        "botsort_reid": PROJECT_ROOT / "configs" / "trackers" / "botsort_reid.yaml"
    }

    if tracker_type in tracker_configs and tracker_configs[tracker_type].exists():
        return str(tracker_configs[tracker_type])
    return None


def _load_calibration(video_id: str, video_db_id: str, calibration_id: str = None):
    """Lädt die Kalibrierung für ein Video."""
    if calibration_id:
        return db.session.get(Calibration, calibration_id)

    # Fallback: Aktive Kalibrierung für dieses Video
    calibration = db.session.query(Calibration).filter_by(
        video_id=video_db_id,
        is_active=True
    ).first()

    # Falls keine aktive, nimm die neueste aktive
    if not calibration:
        calibration = db.session.query(Calibration).filter_by(
            is_active=True
        ).order_by(Calibration.created_at.desc()).first()

    return calibration


# === Game Periods API (Spielzeit-Definitionen) ===

@analysis_bp.route("/api/videos/<video_id>/game-periods", methods=["GET"])
def get_game_periods(video_id):
    """Spielzeiten für ein Video abrufen."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    periods = db.session.query(GamePeriod).filter_by(
        video_id=video.id
    ).order_by(GamePeriod.period_index).all()

    return jsonify({
        "periods": [
            {
                "id": p.id,
                "start": p.start_time,
                "end": p.end_time
            }
            for p in periods
        ]
    })


@analysis_bp.route("/api/videos/<video_id>/game-periods", methods=["POST"])
def save_game_periods(video_id):
    """Spielzeiten für ein Video speichern."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    periods = data.get("periods", [])

    # Alte Spielzeiten löschen
    db.session.query(GamePeriod).filter_by(video_id=video.id).delete()

    # Neue Spielzeiten speichern
    for idx, period in enumerate(periods):
        gp = GamePeriod(
            video_id=video.id,
            start_time=period.get("start", 0),
            end_time=period.get("end", 0),
            period_index=idx
        )
        db.session.add(gp)

    db.session.commit()

    return jsonify({
        "success": True,
        "count": len(periods)
    })


# === Chunk-Tracking API ===

@analysis_bp.route("/api/videos/<video_id>/analysis-chunks", methods=["GET"])
def get_analysis_chunks(video_id):
    """Alle Chunks und Performance-Statistiken für ein Video abrufen."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    chunks = db.session.query(AnalysisChunk).filter_by(
        video_id=video.id
    ).order_by(AnalysisChunk.chunk_index).all()

    # Statistiken berechnen
    completed_chunks = [c for c in chunks if c.status == "completed"]
    total_processing_time = sum(c.processing_time or 0 for c in completed_chunks)
    total_video_time = sum((c.end_time - c.start_time) for c in completed_chunks)

    avg_realtime_factor = 0
    if total_processing_time > 0:
        avg_realtime_factor = total_video_time / total_processing_time

    return jsonify({
        "chunks": [
            {
                "id": str(c.id),
                "chunk_index": c.chunk_index,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "status": c.status,
                "frames_processed": c.frames_processed or 0,
                "detections_count": c.detections_count or 0,
                "processing_time": c.processing_time,
                "error_message": c.error_message
            }
            for c in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "completed_chunks": len(completed_chunks),
            "pending_chunks": len([c for c in chunks if c.status == "pending"]),
            "running_chunks": len([c for c in chunks if c.status == "running"]),
            "error_chunks": len([c for c in chunks if c.status == "error"]),
            "total_processing_time": round(total_processing_time, 2),
            "total_video_time": round(total_video_time, 2),
            "avg_realtime_factor": round(avg_realtime_factor, 2),
            "avg_seconds_per_video_second": round(total_processing_time / total_video_time, 3) if total_video_time > 0 else 0
        }
    })


@analysis_bp.route("/api/videos/<video_id>/analysis-chunks/reset", methods=["POST"])
def reset_analysis_chunks(video_id):
    """Alle Chunks für ein Video zurücksetzen."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Alle Chunks löschen
    db.session.query(AnalysisChunk).filter_by(video_id=video.id).delete()
    db.session.commit()

    # Chunk-Dateien löschen
    chunk_dir = PROJECT_ROOT / "data" / "analysis" / "chunks" / str(video.id)
    if chunk_dir.exists():
        import shutil
        shutil.rmtree(chunk_dir)

    return jsonify({"success": True})


@analysis_bp.route("/api/videos/<video_id>/track-chunk", methods=["POST"])
def track_chunk(video_id):
    """Führt Tracking für einen einzelnen 1-Minuten-Chunk aus."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    chunk_id = data.get("chunk_id", 0)
    start_time = data.get("start", 0)
    end_time = data.get("end", 60)
    model_path = data.get("model_path")
    tracker_config = data.get("tracker_config")  # Pfad zur Tracker-YAML
    conf_threshold = data.get("conf_threshold", 0.5)
    target_fps = data.get("target_fps", 5)
    team_config = data.get("team_config", {})
    enable_team_assign = data.get("enable_team_assign", True)
    enable_jersey_ocr = data.get("enable_jersey_ocr", False)
    team_assign_interval = data.get("team_assign_interval", 10.0)
    jersey_ocr_interval = data.get("jersey_ocr_interval", 5.0)

    # Chunk in DB suchen oder erstellen
    chunk = db.session.query(AnalysisChunk).filter_by(
        video_id=video.id,
        chunk_index=chunk_id
    ).first()

    if not chunk:
        chunk = AnalysisChunk(
            video_id=video.id,
            chunk_index=chunk_id,
            start_time=start_time,
            end_time=end_time,
            status="pending"
        )
        db.session.add(chunk)
        db.session.commit()

    # Falls bereits abgeschlossen, überspringen
    if chunk.status == "completed":
        return jsonify({
            "success": True,
            "skipped": True,
            "message": "Chunk bereits verarbeitet"
        })

    # Falls vorheriger Fehler, zurücksetzen
    if chunk.status == "error":
        chunk.error_message = None

    # Status auf "running" setzen
    chunk.status = "running"
    chunk.started_at = datetime.utcnow()
    db.session.commit()

    import time
    tracking_start_time = time.time()

    try:
        # Modell laden
        if not model_path:
            return jsonify({"error": "Kein Modell angegeben"}), 400

        # Video-Pfad prüfen
        video_path = PROJECT_ROOT / video.file_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

        # YOLO Tracker initialisieren
        tracker = YOLOTracker(
            model_path=model_path,
            tracker_config=tracker_config,
            conf_threshold=conf_threshold
        )

        # TeamAssigner und JerseyOCR initialisieren
        team_assigner = None
        if enable_team_assign:
            team_assigner = TeamAssigner(team_config=team_config)
            if not team_assigner.is_available:
                print("Warning: CLIP model not available for team assignment")
                team_assigner = None

        jersey_detector = None
        if enable_jersey_ocr:
            jersey_detector = JerseyOCR()
            if not jersey_detector.is_available:
                print("Warning: OCR not available for jersey detection")
                jersey_detector = None

        # Spielfeld-Polygon laden (für Filterung)
        field_polygon = _load_field_polygon(video_id, video.id)

        # Frame-Bereich berechnen
        fps = video.fps or 25
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_step = max(1, int(fps / target_fps))

        # Tracking durchführen mit Team-Zuweisung und Jersey-OCR
        results = tracker.track_video_segment(
            video_path=str(video_path),
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            field_polygon=field_polygon,
            team_assigner=team_assigner,
            jersey_detector=jersey_detector,
            team_assign_interval=team_assign_interval,
            jersey_ocr_interval=jersey_ocr_interval
        )

        # Ergebnisse verarbeiten
        frames_processed = len(results.get("frames", []))
        total_detections = sum(
            len(f.get("detections", []))
            for f in results.get("frames", [])
        )

        # Ergebnisse in JSON-Datei speichern
        chunk_dir = PROJECT_ROOT / "data" / "analysis" / "chunks" / str(video.id)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / f"chunk_{chunk_id:04d}.json"

        with open(chunk_file, "w") as f:
            json.dump({
                "video_id": str(video.id),
                "chunk_id": chunk_id,
                "start_time": start_time,
                "end_time": end_time,
                "frames_processed": frames_processed,
                "detections_count": total_detections,
                "results": results
            }, f)

        # Tracking-Zeit berechnen
        tracking_end_time = time.time()
        processing_time = tracking_end_time - tracking_start_time
        chunk_duration = end_time - start_time  # Sekunden Video

        # Chunk als abgeschlossen markieren
        chunk.status = "completed"
        chunk.completed_at = datetime.utcnow()
        chunk.frames_processed = frames_processed
        chunk.detections_count = total_detections
        chunk.processing_time = processing_time
        db.session.commit()

        # Realtime-Faktor berechnen (1.0 = Echtzeit)
        realtime_factor = chunk_duration / processing_time if processing_time > 0 else 0

        return jsonify({
            "success": True,
            "chunk_id": chunk_id,
            "frames_processed": frames_processed,
            "detections_count": total_detections,
            "processing_time": round(processing_time, 2),
            "chunk_duration": round(chunk_duration, 2),
            "realtime_factor": round(realtime_factor, 2),
            "seconds_per_video_second": round(processing_time / chunk_duration, 3) if chunk_duration > 0 else 0
        })

    except Exception as e:
        # Fehler speichern
        chunk.status = "error"
        chunk.error_message = str(e)
        db.session.commit()

        return jsonify({
            "error": str(e),
            "chunk_id": chunk_id
        }), 500


@analysis_bp.route("/api/videos/<video_id>/chunk-positions/<int:chunk_index>")
def get_chunk_positions(video_id, chunk_index):
    """Tracking-Daten für einen einzelnen Chunk mit Spielfeld-Koordinaten abrufen."""
    from src.processing.calibration import load_lens_profile

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Kalibrierung laden
    calibration_id = request.args.get("calibration_id")
    calibration = _load_calibration(video_id, video.id, calibration_id)

    # Position Mapper erstellen (falls Kalibrierung vorhanden)
    position_mapper = None
    if calibration and calibration.calibration_data:
        cal_data = calibration.calibration_data
        video_width = video.width if hasattr(video, 'width') and video.width else 1920
        video_height = video.height if hasattr(video, 'height') and video.height else 1080

        try:
            position_mapper = PositionMapper.from_calibration_data(
                cal_data=cal_data,
                video_width=video_width,
                video_height=video_height,
                lens_profile_loader=load_lens_profile
            )
        except Exception as e:
            print(f"[ChunkPositions] Could not create PositionMapper: {e}")

    # Chunk-Datei laden
    chunk_file = PROJECT_ROOT / "data" / "analysis" / "chunks" / str(video.id) / f"chunk_{chunk_index:04d}.json"

    if not chunk_file.exists():
        return jsonify({"error": f"Chunk {chunk_index} nicht gefunden"}), 404

    try:
        with open(chunk_file, "r") as f:
            chunk_data = json.load(f)

        # Ergebnisse aufbereiten
        frames = []
        results = chunk_data.get("results", {})

        for frame in results.get("frames", []):
            frame_data = {
                "frame_idx": frame.get("frame_idx"),
                "timestamp": frame.get("timestamp", 0) - chunk_data.get("start_time", 0),
                "detections": []
            }

            for det in frame.get("detections", []):
                # Pixel-Koordinaten (Mittelpunkt der BBox)
                center = det.get("center")
                pixel_x = center[0] if center else None
                pixel_y = center[1] if center else None

                # Transformiere zu Spielfeld-Koordinaten
                field_x = None
                field_y = None

                if position_mapper and pixel_x is not None and pixel_y is not None:
                    try:
                        field_pos = position_mapper.pixel_to_field(pixel_x, pixel_y)
                        if field_pos:
                            field_x, field_y = field_pos
                    except Exception:
                        pass

                frame_data["detections"].append({
                    "track_id": det.get("track_id"),
                    "bbox": det.get("bbox"),
                    "confidence": det.get("confidence"),
                    "class_name": det.get("class_name"),
                    "team": det.get("team"),
                    "jersey_number": det.get("jersey_number"),
                    "field_x": field_x,
                    "field_y": field_y,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y
                })

            frames.append(frame_data)

        return jsonify({
            "chunk_index": chunk_index,
            "start_time": chunk_data.get("start_time"),
            "end_time": chunk_data.get("end_time"),
            "has_calibration": position_mapper is not None,
            "frames": frames
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@analysis_bp.route("/api/videos/<video_id>/chunks", methods=["GET"])
def get_chunks_status(video_id):
    """Status aller Chunks für ein Video abrufen."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    chunks = db.session.query(AnalysisChunk).filter_by(
        video_id=video.id
    ).order_by(AnalysisChunk.chunk_index).all()

    # Statistiken berechnen
    completed_chunks = [c for c in chunks if c.status == "completed" and c.processing_time]
    total_processing_time = sum(c.processing_time for c in completed_chunks) if completed_chunks else 0
    total_video_time = sum(c.end_time - c.start_time for c in completed_chunks) if completed_chunks else 0
    avg_realtime_factor = total_video_time / total_processing_time if total_processing_time > 0 else 0

    return jsonify({
        "chunks": [
            {
                "id": c.chunk_index,
                "start": c.start_time,
                "end": c.end_time,
                "status": c.status,
                "frames_processed": c.frames_processed,
                "detections_count": c.detections_count,
                "processing_time": round(c.processing_time, 2) if c.processing_time else None,
                "error": c.error_message
            }
            for c in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "completed_chunks": len(completed_chunks),
            "total_processing_time": round(total_processing_time, 2),
            "total_video_time": round(total_video_time, 2),
            "avg_realtime_factor": round(avg_realtime_factor, 2),
            "avg_seconds_per_video_second": round(total_processing_time / total_video_time, 3) if total_video_time > 0 else 0
        }
    })
