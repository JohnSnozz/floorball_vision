"""
Calibration Routes

Endpoints für Kamera-Kalibrierung und Homography.
"""
import os
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, abort, send_file
from src.web.extensions import db
from src.db.models import Video, Calibration

calibration_bp = Blueprint("calibration", __name__, url_prefix="/calibration")

# Projekt-Root für absolute Pfade (resolve() macht den Pfad absolut)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# === Seiten ===

@calibration_bp.route("/")
def index():
    """Kalibrierungs-Übersicht."""
    videos = db.session.query(Video).filter(Video.status == "ready").all()
    calibrations = db.session.query(Calibration).order_by(Calibration.created_at.desc()).all()

    return render_template(
        "calibration/index.html",
        videos=videos,
        calibrations=calibrations
    )


@calibration_bp.route("/video/<video_id>")
def setup(video_id):
    """Kalibrierungs-Setup für ein Video."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    # Bestehende Kalibrierung laden falls vorhanden
    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    return render_template(
        "calibration/setup.html",
        video=video,
        calibration=calibration
    )


@calibration_bp.route("/lens")
def lens_calibration():
    """Lens-Kalibrierung mit Schachbrett."""
    return render_template("calibration/lens.html")


@calibration_bp.route("/video/<video_id>/quick")
def quick_calibration(video_id):
    """Schnelle Kalibrierung auf entzerrtem Bild."""
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    return render_template(
        "calibration/quick.html",
        video=video,
        calibration=calibration
    )


@calibration_bp.route("/video/<video_id>/simple")
@calibration_bp.route("/video/<video_id>/simple/<calibration_id>")
def simple_calibration(video_id, calibration_id=None):
    """Neue vereinfachte Kalibrierung mit 5-Schritt-Workflow.

    Wenn calibration_id übergeben wird, werden die bestehenden
    Kalibrierungsdaten zum Bearbeiten geladen.
    """
    video = db.session.get(Video, video_id)
    if not video:
        abort(404)

    calibration = None
    if calibration_id:
        # Spezifische Kalibrierung zum Bearbeiten laden
        calibration = db.session.get(Calibration, calibration_id)
        if not calibration or calibration.video_id != video.id:
            abort(404)
    else:
        # Aktive Kalibrierung laden falls vorhanden (für Review-Button)
        calibration = db.session.query(Calibration).filter_by(
            video_id=video.id,
            is_active=True
        ).first()

    # edit_mode zeigt an, ob wir eine bestehende Kalibrierung bearbeiten
    edit_mode = calibration_id is not None

    return render_template(
        "calibration/simple.html",
        video=video,
        calibration=calibration,
        edit_mode=edit_mode
    )


@calibration_bp.route("/lens/checkerboard")
def checkerboard_fullscreen():
    """Fullscreen Schachbrett zum Fotografieren."""
    return render_template("calibration/checkerboard.html")


# === API Endpoints ===

# --- Lens Calibration APIs ---

@calibration_bp.route("/api/lens/detect", methods=["POST"])
def api_detect_checkerboard():
    """
    Erkennt Schachbrett-Ecken in einem Bild.

    Form Data:
        image: Bilddatei
        board_width: Innere Ecken Breite (default: 9)
        board_height: Innere Ecken Höhe (default: 6)
    """
    import cv2
    import numpy as np
    import tempfile

    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild hochgeladen"}), 400

    file = request.files['image']
    board_width = int(request.form.get('board_width', 9))
    board_height = int(request.form.get('board_height', 6))

    try:
        # Bild in numpy array laden
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Bild konnte nicht gelesen werden"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Schachbrett-Ecken finden
        ret, corners = cv2.findChessboardCorners(
            gray,
            (board_width, board_height),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        if ret:
            # Subpixel-Genauigkeit
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            return jsonify({
                "detected": True,
                "corners": corners.reshape(-1, 2).tolist(),
                "image_size": [img.shape[1], img.shape[0]]
            })
        else:
            return jsonify({
                "detected": False,
                "corners": None,
                "image_size": [img.shape[1], img.shape[0]]
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/lens/calibrate", methods=["POST"])
def api_calibrate_lens():
    """
    Führt die Lens-Kalibrierung mit mehreren Schachbrett-Bildern durch.

    Form Data:
        images: Mehrere Bilddateien
        board_width: Innere Ecken Breite
        board_height: Innere Ecken Höhe
        square_size: Quadratgrösse in mm
        lens_type: 'fisheye' oder 'standard'
        camera_name: Name der Kamera
    """
    import cv2
    import numpy as np

    if 'images' not in request.files:
        return jsonify({"error": "Keine Bilder hochgeladen"}), 400

    files = request.files.getlist('images')
    board_width = int(request.form.get('board_width', 9))
    board_height = int(request.form.get('board_height', 6))
    square_size = float(request.form.get('square_size', 30))  # mm
    lens_type = request.form.get('lens_type', 'fisheye')
    camera_name = request.form.get('camera_name', 'Unknown')

    # 3D Punkte für das Schachbrett (in mm)
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D Punkte in der realen Welt
    img_points = []  # 2D Punkte im Bild
    img_size = None
    images_used = 0
    images_total = len(files)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for file in files:
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                continue

            if img_size is None:
                img_size = (img.shape[1], img.shape[0])

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray,
                (board_width, board_height),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)
                images_used += 1

        except Exception as e:
            continue

    if images_used < 3:
        return jsonify({
            "success": False,
            "error": f"Nur {images_used} Bilder mit erkanntem Muster. Mindestens 3 erforderlich."
        }), 400

    try:
        if lens_type == 'fisheye':
            # Fisheye-Kalibrierung mit OpenCV fisheye Modul
            # CALIB_CHECK_COND entfernt - kann bei bestimmten Bildern fehlschlagen
            calibration_flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                cv2.fisheye.CALIB_FIX_SKEW
            )

            # Objekt-Punkte müssen für Fisheye angepasst werden
            obj_points_fish = [op.reshape(1, -1, 3).astype(np.float64) for op in obj_points]
            img_points_fish = [ip.reshape(1, -1, 2).astype(np.float64) for ip in img_points]

            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj_points))]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(obj_points))]

            print(f"Starting fisheye calibration with {len(obj_points)} images, size={img_size}")

            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_points_fish,
                img_points_fish,
                img_size,
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

            print(f"Fisheye calibration done: error={ret}, K={K}, D={D.flatten()}")

            camera_matrix = K.tolist()
            distortion_coeffs = D.flatten().tolist()

        else:
            # Standard-Kalibrierung
            print(f"Starting standard calibration with {len(obj_points)} images, size={img_size}")

            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                img_size,
                None,
                None
            )

            print(f"Standard calibration done: error={ret}")

            camera_matrix = K.tolist()
            distortion_coeffs = dist.flatten().tolist()

        return jsonify({
            "success": True,
            "reprojection_error": float(ret),
            "camera_matrix": camera_matrix,
            "distortion_coeffs": distortion_coeffs,
            "image_size": list(img_size),
            "images_used": images_used,
            "images_total": images_total,
            "lens_type": lens_type,
            "camera_name": camera_name,
            "board_size": [board_width, board_height],
            "square_size_mm": square_size
        })

    except cv2.error as cv_err:
        print(f"OpenCV error during calibration: {cv_err}")
        return jsonify({
            "success": False,
            "error": f"OpenCV Fehler: {str(cv_err)}. Versuche mehr oder bessere Bilder zu verwenden."
        }), 500

    except Exception as e:
        import traceback
        print(f"Calibration error: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --- Temporäre Bild-Speicherung für Lens-Kalibrierung ---

# In-Memory Storage für temporäre Kalibrierungsbilder (Session-basiert)
# In Produktion sollte das in Redis oder einer DB gespeichert werden
_temp_calibration_images = {}


@calibration_bp.route("/api/lens/temp-images", methods=["GET"])
def api_get_temp_images():
    """Gibt alle temporär gespeicherten Kalibrierungsbilder zurück."""
    from flask import session

    session_id = session.get('_id', 'default')
    images = _temp_calibration_images.get(session_id, [])

    return jsonify({
        "images": images,
        "count": len(images)
    })


@calibration_bp.route("/api/lens/temp-images", methods=["POST"])
def api_save_temp_image():
    """Speichert ein Kalibrierungsbild temporär."""
    from flask import session

    data = request.get_json()
    if not data or 'id' not in data or 'dataUrl' not in data:
        return jsonify({"error": "id und dataUrl erforderlich"}), 400

    session_id = session.get('_id', 'default')

    if session_id not in _temp_calibration_images:
        _temp_calibration_images[session_id] = []

    # Prüfen ob Bild schon existiert (Update)
    existing_idx = None
    for i, img in enumerate(_temp_calibration_images[session_id]):
        if img['id'] == data['id']:
            existing_idx = i
            break

    image_data = {
        "id": data['id'],
        "name": data.get('name', 'unknown'),
        "dataUrl": data['dataUrl'],
        "detected": data.get('detected'),
        "corners": data.get('corners')
    }

    if existing_idx is not None:
        _temp_calibration_images[session_id][existing_idx] = image_data
    else:
        _temp_calibration_images[session_id].append(image_data)

    return jsonify({"success": True, "count": len(_temp_calibration_images[session_id])})


@calibration_bp.route("/api/lens/temp-images/<image_id>", methods=["DELETE"])
def api_delete_temp_image(image_id):
    """Löscht ein temporäres Kalibrierungsbild."""
    from flask import session

    session_id = session.get('_id', 'default')

    if session_id in _temp_calibration_images:
        _temp_calibration_images[session_id] = [
            img for img in _temp_calibration_images[session_id]
            if img['id'] != image_id
        ]

    return jsonify({"success": True})


@calibration_bp.route("/api/lens/temp-images", methods=["DELETE"])
def api_clear_temp_images():
    """Löscht alle temporären Kalibrierungsbilder."""
    from flask import session

    session_id = session.get('_id', 'default')

    if session_id in _temp_calibration_images:
        del _temp_calibration_images[session_id]

    return jsonify({"success": True})


@calibration_bp.route("/api/lens/profiles", methods=["GET"])
def api_get_lens_profiles():
    """Gibt alle gespeicherten Lens-Profile zurück."""
    import json

    profiles_dir = PROJECT_ROOT / "data" / "lens_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profiles = []
    for file in profiles_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                profile = json.load(f)
                profile['id'] = file.stem
                profiles.append(profile)
        except:
            continue

    return jsonify(profiles)


@calibration_bp.route("/api/lens/profiles", methods=["POST"])
def api_save_lens_profile():
    """Speichert ein neues Lens-Profil."""
    import json
    import uuid
    from datetime import datetime

    data = request.get_json()

    if not data or 'name' not in data or 'calibration_data' not in data:
        return jsonify({"error": "Name und calibration_data erforderlich"}), 400

    cal_data = data['calibration_data']

    profile = {
        "name": data['name'],
        "camera_name": data.get('camera_name', 'Unknown'),
        "lens_type": data.get('lens_type', 'fisheye'),
        "resolution": {
            "width": cal_data.get('image_size', [1920, 1080])[0],
            "height": cal_data.get('image_size', [1920, 1080])[1]
        },
        "fisheye_params": {
            "camera_matrix": cal_data['camera_matrix'],
            "distortion_coeffs": cal_data['distortion_coeffs']
        },
        "reprojection_error": cal_data.get('reprojection_error'),
        "board_size": cal_data.get('board_size'),
        "square_size_mm": cal_data.get('square_size_mm'),
        "created_at": datetime.utcnow().isoformat(),
        "source": "Schachbrett-Kalibrierung"
    }

    profiles_dir = PROJECT_ROOT / "data" / "lens_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Eindeutiger Dateiname
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in data['name'])
    profile_path = profiles_dir / f"{safe_name}.json"

    # Falls existiert, Nummer anhängen
    counter = 1
    while profile_path.exists():
        profile_path = profiles_dir / f"{safe_name}_{counter}.json"
        counter += 1

    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    return jsonify({
        "success": True,
        "id": profile_path.stem,
        "path": str(profile_path)
    })


@calibration_bp.route("/api/lens/profiles/<profile_id>", methods=["DELETE"])
def api_delete_lens_profile(profile_id):
    """Löscht ein Lens-Profil."""
    profiles_dir = PROJECT_ROOT / "data" / "lens_profiles"
    profile_path = profiles_dir / f"{profile_id}.json"

    if profile_path.exists():
        profile_path.unlink()
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Profil nicht gefunden"}), 404


# --- Video Calibration APIs ---

@calibration_bp.route("/api/videos/<video_id>/screenshot", methods=["POST"])
def api_extract_screenshot(video_id):
    """
    Extrahiert einen Screenshot aus dem Video für die Kalibrierung.

    JSON Body:
        timestamp: Zeit in Sekunden (default: 10)
    """
    from src.processing.calibration import extract_calibration_frame

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    if not video.file_path or not os.path.exists(video.file_path):
        return jsonify({"error": "Video-Datei nicht gefunden"}), 404

    data = request.get_json() or {}
    timestamp = data.get("timestamp", 10.0)

    try:
        # Screenshot extrahieren (absoluter Pfad)
        output_dir = PROJECT_ROOT / "data" / "calibration" / video_id
        screenshot_path = extract_calibration_frame(
            video_path=video.file_path,
            timestamp=timestamp,
            output_dir=str(output_dir)
        )

        return jsonify({
            "success": True,
            "screenshot_path": screenshot_path,
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/screenshots", methods=["POST"])
def api_extract_screenshots(video_id):
    """
    Extrahiert mehrere Screenshots aus dem Video für die Kalibrierung.

    JSON Body:
        num_screenshots: Anzahl Screenshots (default: 20)
                        Bei 0: Nur existierende Screenshots laden (keine neuen erstellen)
    """
    from src.processing.calibration import extract_multiple_screenshots, get_existing_screenshots

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    num_screenshots = data.get("num_screenshots", 20)

    # Absoluter Pfad für Screenshots
    output_dir = PROJECT_ROOT / "data" / "calibration" / video_id / "screenshots"

    # Bei num_screenshots=0: Nur existierende Screenshots laden
    if num_screenshots == 0:
        existing = get_existing_screenshots(str(output_dir))
        return jsonify({
            "success": True,
            "screenshots": existing,
            "count": len(existing)
        })

    if not video.file_path or not os.path.exists(video.file_path):
        return jsonify({"error": "Video-Datei nicht gefunden"}), 404

    try:
        screenshots = extract_multiple_screenshots(
            video_path=video.file_path,
            output_dir=str(output_dir),
            num_screenshots=num_screenshots
        )

        return jsonify({
            "success": True,
            "screenshots": screenshots,
            "count": len(screenshots)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration", methods=["GET"])
def api_get_calibration(video_id):
    """Gibt die aktive Kalibrierung für ein Video zurück."""
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        return jsonify({"calibration": None})

    return jsonify({
        "calibration": {
            "id": str(calibration.id),
            "name": calibration.name,
            "calibration_data": calibration.calibration_data,
            "current_step": calibration.current_step,
            "field_length": calibration.field_length,
            "field_width": calibration.field_width,
            "created_at": calibration.created_at.isoformat() if calibration.created_at else None,
            "updated_at": calibration.updated_at.isoformat() if calibration.updated_at else None
        }
    })


@calibration_bp.route("/api/videos/<video_id>/calibration/step/<int:step>", methods=["POST"])
def api_save_calibration_step(video_id, step):
    """
    Speichert einen einzelnen Kalibrierungsschritt.

    Step 1: Screenshot-Auswahl
    JSON Body: { "screenshot_index": 0 }

    Step 2: Spielfeld-Begrenzung (Boundary)
    JSON Body: { "boundary_polygon": [[x1,y1], [x2,y2], ...] }

    Step 3: Punkte-Zuordnung
    JSON Body: {
        "image_points": [[x1,y1], ...],
        "field_points": [[x1,y1], ...],
        "straight_lines": [{"points": [...], "type": "horizontal"}, ...]
    }

    Step 4: Verifizierung
    JSON Body: {
        "lens_profile_id": "gopro_hero10_...",
        "verified": true
    }
    """
    from datetime import datetime
    from src.processing.calibration import calculate_homography

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    if step < 1 or step > 4:
        return jsonify({"error": "Ungültiger Schritt (1-4)"}), 400

    data = request.get_json() or {}

    # Bestehende Kalibrierung laden oder neue erstellen
    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        calibration = Calibration(
            video_id=video.id,
            name=f"Kalibrierung {video.filename}",
            calibration_data={},
            is_active=True
        )
        db.session.add(calibration)

    # Calibration_data als Dict sicherstellen
    if not calibration.calibration_data:
        calibration.calibration_data = {}

    cal_data = dict(calibration.calibration_data)  # Kopie erstellen für Modifikation

    try:
        timestamp = datetime.utcnow().isoformat()

        if step == 1:
            # Screenshot-Auswahl
            cal_data["step1_screenshots"] = {
                "screenshot_index": data.get("screenshot_index", 0),
                "generated_at": timestamp
            }

        elif step == 2:
            # Spielfeld-Begrenzung
            boundary = data.get("boundary_polygon", [])
            cal_data["step2_boundary"] = {
                "boundary_polygon": boundary,
                "saved_at": timestamp
            }

        elif step == 3:
            # Punkte-Zuordnung
            image_points = data.get("image_points", [])
            field_points = data.get("field_points", [])
            straight_lines = data.get("straight_lines", [])

            cal_data["step3_points"] = {
                "image_points": image_points,
                "field_points": field_points,
                "straight_lines": straight_lines,
                "saved_at": timestamp
            }

            # Homographie berechnen wenn genug Punkte vorhanden
            if len(image_points) >= 4 and len(image_points) == len(field_points):
                homography_matrix = calculate_homography(
                    image_points=image_points,
                    field_points=field_points
                )
                cal_data["step3_points"]["homography_matrix"] = (
                    homography_matrix.tolist() if hasattr(homography_matrix, 'tolist')
                    else homography_matrix
                )

        elif step == 4:
            # Verifizierung
            cal_data["step4_verify"] = {
                "lens_profile_id": data.get("lens_profile_id"),
                "verified": data.get("verified", False),
                "verified_at": timestamp
            }

            # Finale Homographie-Matrix speichern (falls noch nicht vorhanden)
            if "step3_points" in cal_data and "homography_matrix" in cal_data["step3_points"]:
                cal_data["step4_verify"]["homography_matrix"] = cal_data["step3_points"]["homography_matrix"]

        # Kalibrierung aktualisieren
        calibration.calibration_data = cal_data
        calibration.current_step = step
        db.session.commit()

        return jsonify({
            "success": True,
            "calibration_id": str(calibration.id),
            "current_step": calibration.current_step,
            "step_data": cal_data.get(f"step{step}_screenshots") or
                        cal_data.get(f"step{step}_boundary") or
                        cal_data.get(f"step{step}_points") or
                        cal_data.get(f"step{step}_verify")
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration/step/<int:step>", methods=["GET"])
def api_get_calibration_step(video_id, step):
    """
    Lädt die Daten eines einzelnen Kalibrierungsschritts.

    Returns für jeden Schritt die entsprechenden Daten oder null.
    """
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    if step < 1 or step > 4:
        return jsonify({"error": "Ungültiger Schritt (1-4)"}), 400

    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        return jsonify({"step_data": None, "current_step": 0})

    cal_data = calibration.calibration_data or {}

    step_key_map = {
        1: "step1_screenshots",
        2: "step2_boundary",
        3: "step3_points",
        4: "step4_verify"
    }

    step_data = cal_data.get(step_key_map[step])

    return jsonify({
        "step_data": step_data,
        "current_step": calibration.current_step,
        "calibration_id": str(calibration.id)
    })


@calibration_bp.route("/api/videos/<video_id>/calibration/optimize", methods=["POST"])
def api_optimize_calibration(video_id):
    """
    Optimiert gleichzeitig Fisheye-Korrektur UND Homographie aus den Referenzpunkten.

    Diese Funktion verwendet alle gesetzten Referenzpunkte, um sowohl die radiale
    Verzerrung (k1) als auch die perspektivische Transformation zu berechnen.

    JSON Body:
        image_width: Bildbreite in Pixel
        image_height: Bildhöhe in Pixel
        optimize_distortion: Ob k1 optimiert werden soll (default: true)

    Returns:
        homography_matrix, k1, reprojection_error, etc.
    """
    from src.processing.calibration import calculate_combined_calibration

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    # Aktive Kalibrierung laden
    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        return jsonify({"error": "Keine aktive Kalibrierung gefunden"}), 404

    cal_data = calibration.calibration_data or {}

    # Punkte aus step3_points laden
    step3 = cal_data.get("step3_points", {})
    image_points = step3.get("image_points", [])
    field_points = step3.get("field_points", [])

    if len(image_points) < 4:
        return jsonify({"error": "Mindestens 4 Referenzpunkte erforderlich"}), 400

    if len(image_points) != len(field_points):
        return jsonify({"error": "Anzahl Bild-Punkte und Feld-Punkte muss gleich sein"}), 400

    data = request.get_json() or {}
    image_width = data.get("image_width", video.width or 1920)
    image_height = data.get("image_height", video.height or 1080)
    optimize_distortion = data.get("optimize_distortion", True)

    try:
        # Kombinierte Optimierung durchführen
        result = calculate_combined_calibration(
            image_points=image_points,
            field_points=field_points,
            image_size=(image_width, image_height),
            optimize_distortion=optimize_distortion
        )

        # Ergebnis in Kalibrierung speichern
        cal_data["combined_calibration"] = {
            "homography_matrix": result["homography_matrix"],
            "k1": result["k1"],
            "focal_length": result["focal_length"],
            "camera_matrix": result["camera_matrix"],
            "distortion_coeffs": result["distortion_coeffs"],
            "image_center": result["image_center"],
            "reprojection_error": result["reprojection_error"],
            "max_error": result["max_error"],
            "image_size": [image_width, image_height],
            "optimized_at": __import__('datetime').datetime.utcnow().isoformat()
        }

        # Auch die alte homography_matrix aktualisieren für Kompatibilität
        cal_data["homography_matrix"] = result["homography_matrix"]
        if "step3_points" in cal_data:
            cal_data["step3_points"]["homography_matrix"] = result["homography_matrix"]

        # JSONB-Feld als geändert markieren damit SQLAlchemy es speichert
        from sqlalchemy.orm.attributes import flag_modified
        calibration.calibration_data = cal_data
        flag_modified(calibration, "calibration_data")
        db.session.commit()

        return jsonify({
            "success": True,
            "homography_matrix": result["homography_matrix"],
            "k1": result["k1"],
            "focal_length": result["focal_length"],
            "reprojection_error": result["reprojection_error"],
            "max_error": result["max_error"],
            "point_errors": result["point_errors"],
            "message": f"Optimierung erfolgreich. k1={result['k1']:.4f}, Fehler={result['reprojection_error']:.3f}m"
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration", methods=["POST"])
def api_save_calibration(video_id):
    """
    Speichert eine neue Kalibrierung.

    JSON Body:
        name: Name der Kalibrierung
        boundary_polygon: Liste von [x, y] Punkten (Pixel) für Banden-Polygon
        image_points: Liste von [x, y] Punkten (Pixel) im Bild
        field_points: Liste von [x, y] Punkten (Meter) auf dem Spielfeld
        field_length: Spielfeld-Länge in Metern (default: 40)
        field_width: Spielfeld-Breite in Metern (default: 20)
    """
    from src.processing.calibration import calculate_homography

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json()

    # Validierung
    if not data.get("image_points") or not data.get("field_points"):
        return jsonify({"error": "image_points und field_points sind erforderlich"}), 400

    if len(data["image_points"]) != len(data["field_points"]):
        return jsonify({"error": "Anzahl image_points und field_points muss gleich sein"}), 400

    if len(data["image_points"]) < 4:
        return jsonify({"error": "Mindestens 4 Punkt-Paare erforderlich"}), 400

    try:
        # Homography-Matrix berechnen
        homography_matrix = calculate_homography(
            image_points=data["image_points"],
            field_points=data["field_points"]
        )

        # Alte Kalibrierungen deaktivieren
        db.session.query(Calibration).filter_by(
            video_id=video.id,
            is_active=True
        ).update({"is_active": False})

        # Neue Kalibrierung speichern
        calibration_data = {
            "boundary_polygon": data.get("boundary_polygon", []),
            "image_points": data["image_points"],
            "field_points": data["field_points"],
            "homography_matrix": homography_matrix.tolist() if hasattr(homography_matrix, 'tolist') else homography_matrix
        }

        calibration = Calibration(
            video_id=video.id,
            name=data.get("name", f"Kalibrierung {video.filename}"),
            calibration_data=calibration_data,
            field_length=data.get("field_length", 40.0),
            field_width=data.get("field_width", 20.0),
            is_active=True
        )

        db.session.add(calibration)
        db.session.commit()

        return jsonify({
            "success": True,
            "calibration_id": str(calibration.id),
            "homography_matrix": calibration_data["homography_matrix"]
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration/test", methods=["POST"])
def api_test_calibration(video_id):
    """
    Generiert ein Test-Overlay um die Kalibrierung zu verifizieren.

    Zeichnet ein Grid auf den Screenshot basierend auf der Homography.
    Bei Fisheye-Profil werden gekrümmte Linien gezeichnet.

    JSON Body:
        screenshot_path: Pfad zum Screenshot (optional)
        lens_profile_id: ID des Lens-Profils für Fisheye-Korrektur (optional)
    """
    from src.processing.calibration import generate_test_overlay, load_lens_profile

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        return jsonify({"error": "Keine aktive Kalibrierung gefunden"}), 404

    data = request.get_json() or {}
    screenshot_path = data.get("screenshot_path")
    lens_profile_id = data.get("lens_profile_id")

    # Lens-Profil laden falls angegeben
    lens_profile = None
    if lens_profile_id:
        lens_profile = load_lens_profile(lens_profile_id)

    # Absolute Pfade für Screenshot und Overlay
    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id

    if not screenshot_path or not os.path.exists(screenshot_path):
        # Versuche Standard-Screenshot zu finden
        screenshot_path = str(calibration_dir / "calibration_frame.jpg")
        if not os.path.exists(screenshot_path):
            return jsonify({"error": "Screenshot nicht gefunden"}), 404

    try:
        overlay_path = generate_test_overlay(
            screenshot_path=screenshot_path,
            calibration_data=calibration.calibration_data,
            output_path=str(calibration_dir / "test_overlay.jpg"),
            lens_profile=lens_profile
        )

        return jsonify({
            "success": True,
            "overlay_path": overlay_path,
            "lens_profile_used": lens_profile_id if lens_profile else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/calibration/<calibration_id>/transform", methods=["POST"])
def api_transform_points(calibration_id):
    """
    Transformiert Pixel-Koordinaten zu Feld-Koordinaten.

    JSON Body:
        points: Liste von [x, y] Pixel-Koordinaten

    Returns:
        field_points: Liste von [x, y] Feld-Koordinaten (Meter)
        inside_field: Liste von Booleans (ob Punkt innerhalb Bande)
    """
    from src.processing.calibration import transform_points, points_inside_polygon

    calibration = db.session.get(Calibration, calibration_id)
    if not calibration:
        return jsonify({"error": "Kalibrierung nicht gefunden"}), 404

    data = request.get_json()
    points = data.get("points", [])

    if not points:
        return jsonify({"error": "Keine Punkte angegeben"}), 400

    try:
        cal_data = calibration.calibration_data

        # Punkte transformieren
        field_points = transform_points(
            points=points,
            homography_matrix=cal_data["homography_matrix"]
        )

        # Prüfen ob innerhalb Bande
        inside_field = points_inside_polygon(
            points=points,
            polygon=cal_data.get("boundary_polygon", [])
        )

        return jsonify({
            "field_points": field_points,
            "inside_field": inside_field
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/calibrations")
def api_list_calibrations():
    """Listet alle Kalibrierungen."""
    calibrations = db.session.query(Calibration).order_by(Calibration.created_at.desc()).all()

    return jsonify([{
        "id": str(c.id),
        "video_id": str(c.video_id),
        "name": c.name,
        "is_active": c.is_active,
        "field_length": c.field_length,
        "field_width": c.field_width,
        "created_at": c.created_at.isoformat() if c.created_at else None
    } for c in calibrations])


@calibration_bp.route("/api/calibrations/<calibration_id>", methods=["DELETE"])
def api_delete_calibration(calibration_id):
    """Löscht eine Kalibrierung."""
    calibration = db.session.get(Calibration, calibration_id)
    if not calibration:
        return jsonify({"error": "Kalibrierung nicht gefunden"}), 404

    db.session.delete(calibration)
    db.session.commit()

    return jsonify({"success": True})


@calibration_bp.route("/api/calibrations/<calibration_id>/activate", methods=["POST"])
def api_activate_calibration(calibration_id):
    """Aktiviert eine Kalibrierung und deaktiviert alle anderen für dieses Video."""
    calibration = db.session.get(Calibration, calibration_id)
    if not calibration:
        return jsonify({"error": "Kalibrierung nicht gefunden"}), 404

    # Alle anderen Kalibrierungen für dieses Video deaktivieren
    db.session.query(Calibration).filter(
        Calibration.video_id == calibration.video_id,
        Calibration.id != calibration.id
    ).update({"is_active": False})

    # Diese Kalibrierung aktivieren
    calibration.is_active = True
    db.session.commit()

    return jsonify({
        "success": True,
        "calibration_id": str(calibration.id),
        "name": calibration.name
    })


@calibration_bp.route("/api/calibrations/<calibration_id>/rename", methods=["POST"])
def api_rename_calibration(calibration_id):
    """Benennt eine Kalibrierung um."""
    calibration = db.session.get(Calibration, calibration_id)
    if not calibration:
        return jsonify({"error": "Kalibrierung nicht gefunden"}), 404

    data = request.get_json() or {}
    new_name = data.get("name", "").strip()

    calibration.name = new_name if new_name else None
    db.session.commit()

    return jsonify({
        "success": True,
        "calibration_id": str(calibration.id),
        "name": calibration.name
    })


# === Lens Profile API ===

@calibration_bp.route("/api/lens-profiles")
def api_list_lens_profiles():
    """Listet alle verfügbaren Lens-Profile."""
    from src.processing.calibration import get_available_lens_profiles

    profiles = get_available_lens_profiles()
    return jsonify(profiles)


@calibration_bp.route("/api/videos/<video_id>/undistort-screenshot", methods=["POST"])
def api_undistort_screenshot(video_id):
    """
    Entzerrt einen Screenshot mit einem Lens-Profil.

    JSON Body:
        lens_profile_id: ID des Lens-Profils
        screenshot_path: Pfad zum Original-Screenshot (optional)
        balance: 0.0-1.0, wie viel vom Originalbild erhalten bleibt (default: 1.0)
        zoom_out: Faktor > 1.0 vergrößert das Ausgabebild um Ecken zu zeigen (default: 1.2)

    Returns:
        undistorted_path: Pfad zum entzerrten Bild
    """
    import cv2
    import numpy as np
    from src.processing.calibration import load_lens_profile, undistort_image, scale_camera_matrix

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    lens_profile_id = data.get("lens_profile_id")
    balance = data.get("balance", 1.0)  # Default 1.0 um alles zu behalten
    zoom_out = data.get("zoom_out", 1.2)  # Default 1.2 um Ecken zu zeigen

    if not lens_profile_id:
        return jsonify({"error": "lens_profile_id erforderlich"}), 400

    # Lens-Profil laden
    profile = load_lens_profile(lens_profile_id)
    if not profile:
        return jsonify({"error": f"Lens-Profil '{lens_profile_id}' nicht gefunden"}), 404

    # Screenshot-Pfad bestimmen
    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id
    screenshot_path = data.get("screenshot_path")

    if not screenshot_path:
        # Standard-Screenshot suchen
        for candidate in ["calibration_frame.jpg", "screenshots/frame_0000.jpg"]:
            candidate_path = calibration_dir / candidate
            if candidate_path.exists():
                screenshot_path = str(candidate_path)
                break

    if not screenshot_path or not os.path.exists(screenshot_path):
        return jsonify({"error": "Screenshot nicht gefunden"}), 404

    try:
        # Bild laden
        img = cv2.imread(screenshot_path)
        if img is None:
            return jsonify({"error": "Screenshot konnte nicht geladen werden"}), 500

        h, w = img.shape[:2]

        # Kameramatrix und Distortion aus Profil extrahieren
        fisheye_params = profile.get("fisheye_params", {})
        camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
        distortion_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64)

        # Distortion-Koeffizienten als (4,1) Spaltenvektor für OpenCV fisheye
        if distortion_coeffs.ndim == 1:
            distortion_coeffs = distortion_coeffs.reshape(4, 1)

        # Kameramatrix skalieren falls nötig
        profile_res = profile.get("resolution", {})
        profile_w = profile_res.get("width", w)
        profile_h = profile_res.get("height", h)

        if profile_w != w or profile_h != h:
            camera_matrix = scale_camera_matrix(
                camera_matrix,
                (profile_w, profile_h),
                (w, h)
            )

        # Prüfen ob es ein Custom-Profil ist (manuelle Anpassung oder geschätzt)
        # Profile mit "Manuelle Anpassung" oder "estimated: true" verwenden die direkte Methode
        is_custom_profile = (
            profile.get("source") == "Manuelle Anpassung" or
            profile.get("estimated") == True
        )

        # Rotation aus Request lesen (falls übergeben)
        rotation = data.get("rotation", 0.0)

        if is_custom_profile:
            # Custom-Profile: zoom_out und rotation aus Profil lesen (falls nicht überschrieben)
            undistort_params = profile.get("undistort_params", {})
            profile_zoom = undistort_params.get("zoom_out", 1.2)
            profile_rotation = undistort_params.get("rotation", 0.0)

            # Request-Parameter überschreiben Profil-Parameter nur wenn explizit angegeben
            effective_zoom = zoom_out if zoom_out != 1.2 else profile_zoom
            # Rotation: Request-Wert hat Vorrang, sonst Profil-Wert
            effective_rotation = rotation if rotation != 0.0 else profile_rotation

            # Custom-Profile verwenden die gleiche Methode wie beim Erstellen
            # Zoom durch Anpassung der Brennweite, Bildgrösse bleibt gleich
            new_K = camera_matrix.copy()
            new_K[0, 0] /= effective_zoom
            new_K[1, 1] /= effective_zoom

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix, distortion_coeffs,
                np.eye(3), new_K,
                (w, h), cv2.CV_16SC2
            )

            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # Rotation anwenden (falls nicht 0)
            if abs(effective_rotation) > 0.01:
                center = (w / 2, h / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, effective_rotation, 1.0)
                undistorted = cv2.warpAffine(undistorted, rotation_matrix, (w, h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))

            undist_h, undist_w = h, w
            zoom_out = effective_zoom  # Für Response
        else:
            # Standard-Profile verwenden die balance-Methode
            undistorted = undistort_image(
                img,
                camera_matrix,
                distortion_coeffs,
                balance=balance,
                zoom_out=zoom_out
            )
            undist_h, undist_w = undistorted.shape[:2]

        # Speichern
        output_path = calibration_dir / "undistorted_frame.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), undistorted)

        return jsonify({
            "success": True,
            "undistorted_path": str(output_path),
            "original_path": screenshot_path,
            "lens_profile": lens_profile_id,
            "original_size": [w, h],
            "undistorted_size": [undist_w, undist_h],
            "zoom_out": zoom_out,
            "balance": balance,
            "is_custom_profile": is_custom_profile
        })

    except Exception as e:
        import traceback
        print(f"Undistort error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration/transform-points", methods=["POST"])
def api_transform_calibration_points(video_id):
    """
    Transformiert die bestehenden Kalibrierungspunkte mit Lens-Korrektur.
    (Option A: Bestehende Punkte automatisch umrechnen)

    JSON Body:
        lens_profile_id: ID des Lens-Profils

    Returns:
        original_points: Original Bildpunkte
        undistorted_points: Entzerrte Bildpunkte
    """
    import numpy as np
    from src.processing.calibration import load_lens_profile, undistort_points, scale_camera_matrix

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    lens_profile_id = data.get("lens_profile_id")

    if not lens_profile_id:
        return jsonify({"error": "lens_profile_id erforderlich"}), 400

    # Aktive Kalibrierung laden
    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        return jsonify({"error": "Keine aktive Kalibrierung gefunden"}), 404

    cal_data = calibration.calibration_data or {}
    step3 = cal_data.get("step3_points", {})
    image_points = step3.get("image_points", [])

    if not image_points:
        return jsonify({"error": "Keine Bildpunkte in der Kalibrierung"}), 400

    # Lens-Profil laden
    profile = load_lens_profile(lens_profile_id)
    if not profile:
        return jsonify({"error": f"Lens-Profil '{lens_profile_id}' nicht gefunden"}), 404

    try:
        # Bildgrösse
        img_w = video.width or 1920
        img_h = video.height or 1080

        # Kameramatrix und Distortion
        fisheye_params = profile.get("fisheye_params", {})
        camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
        distortion_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64)

        # Skalieren falls nötig
        profile_res = profile.get("resolution", {})
        profile_w = profile_res.get("width", img_w)
        profile_h = profile_res.get("height", img_h)

        if profile_w != img_w or profile_h != img_h:
            camera_matrix = scale_camera_matrix(
                camera_matrix,
                (profile_w, profile_h),
                (img_w, img_h)
            )

        # Punkte transformieren
        pts = np.array(image_points, dtype=np.float64)
        undistorted_pts = undistort_points(pts, camera_matrix, distortion_coeffs)

        return jsonify({
            "success": True,
            "original_points": image_points,
            "undistorted_points": undistorted_pts.tolist(),
            "lens_profile": lens_profile_id
        })

    except Exception as e:
        import traceback
        print(f"Transform points error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration/apply-lens-correction", methods=["POST"])
def api_apply_lens_correction(video_id):
    """
    Wendet Lens-Korrektur auf die Kalibrierung an und berechnet neue Homographie.
    (Option A komplett: Punkte transformieren + Homographie neu berechnen)

    JSON Body:
        lens_profile_id: ID des Lens-Profils

    Returns:
        Neue Homographie-Matrix und Fehlermetriken
    """
    import numpy as np
    from datetime import datetime
    from sqlalchemy.orm.attributes import flag_modified
    from src.processing.calibration import (
        load_lens_profile, undistort_points, scale_camera_matrix, calculate_homography
    )

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    lens_profile_id = data.get("lens_profile_id")

    if not lens_profile_id:
        return jsonify({"error": "lens_profile_id erforderlich"}), 400

    # Aktive Kalibrierung laden
    calibration = db.session.query(Calibration).filter_by(
        video_id=video.id,
        is_active=True
    ).first()

    if not calibration:
        return jsonify({"error": "Keine aktive Kalibrierung gefunden"}), 404

    cal_data = calibration.calibration_data or {}
    step3 = cal_data.get("step3_points", {})
    image_points = step3.get("image_points", [])
    field_points = step3.get("field_points", [])

    if len(image_points) < 4:
        return jsonify({"error": "Mindestens 4 Bildpunkte erforderlich"}), 400

    # Lens-Profil laden
    profile = load_lens_profile(lens_profile_id)
    if not profile:
        return jsonify({"error": f"Lens-Profil '{lens_profile_id}' nicht gefunden"}), 404

    try:
        # Bildgrösse
        img_w = video.width or 1920
        img_h = video.height or 1080

        # Kameramatrix und Distortion
        fisheye_params = profile.get("fisheye_params", {})
        camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
        distortion_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64)

        # Skalieren falls nötig
        profile_res = profile.get("resolution", {})
        profile_w = profile_res.get("width", img_w)
        profile_h = profile_res.get("height", img_h)

        if profile_w != img_w or profile_h != img_h:
            camera_matrix = scale_camera_matrix(
                camera_matrix,
                (profile_w, profile_h),
                (img_w, img_h)
            )

        # Bildpunkte entzerren
        pts = np.array(image_points, dtype=np.float64)
        undistorted_pts = undistort_points(pts, camera_matrix, distortion_coeffs)

        # Neue Homographie berechnen mit entzerrten Punkten
        homography_matrix = calculate_homography(
            image_points=undistorted_pts.tolist(),
            field_points=field_points
        )

        # In Kalibrierung speichern
        cal_data["lens_correction"] = {
            "lens_profile_id": lens_profile_id,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coeffs": distortion_coeffs.tolist(),
            "original_image_points": image_points,
            "undistorted_image_points": undistorted_pts.tolist(),
            "applied_at": datetime.utcnow().isoformat()
        }

        # Homographie aktualisieren
        cal_data["homography_matrix"] = homography_matrix.tolist() if hasattr(homography_matrix, 'tolist') else homography_matrix
        if "step3_points" in cal_data:
            cal_data["step3_points"]["homography_matrix"] = cal_data["homography_matrix"]

        calibration.calibration_data = cal_data
        flag_modified(calibration, "calibration_data")
        db.session.commit()

        return jsonify({
            "success": True,
            "homography_matrix": cal_data["homography_matrix"],
            "lens_profile": lens_profile_id,
            "original_points": image_points,
            "undistorted_points": undistorted_pts.tolist(),
            "message": f"Lens-Korrektur mit '{lens_profile_id}' angewendet"
        })

    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"Apply lens correction error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/lens-profiles/<profile_id>")
def api_get_lens_profile(profile_id):
    """Gibt ein spezifisches Lens-Profil zurück."""
    from src.processing.calibration import load_lens_profile

    profile = load_lens_profile(profile_id)
    if not profile:
        return jsonify({"error": "Lens-Profil nicht gefunden"}), 404

    return jsonify(profile)


@calibration_bp.route("/api/lens/test-undistort", methods=["POST"])
def api_test_undistort():
    """
    Testet ein Lens-Profil mit einem benutzerdefinierten Bild.
    Gibt das entzerrte Bild als Base64 zurück.

    JSON Body:
        image_data: Base64-kodiertes Bild (data:image/...)
        lens_profile_id: ID des Lens-Profils
        balance: 0.0-1.0 (default: 0.5)

    Returns:
        image_data: Base64-kodiertes entzerrtes Bild
    """
    import cv2
    import numpy as np
    import base64
    from src.processing.calibration import load_lens_profile, undistort_image, scale_camera_matrix

    data = request.get_json() or {}
    image_data = data.get("image_data")
    lens_profile_id = data.get("lens_profile_id")
    balance = data.get("balance", 0.5)

    if not image_data:
        return jsonify({"error": "image_data erforderlich"}), 400

    if not lens_profile_id:
        return jsonify({"error": "lens_profile_id erforderlich"}), 400

    # Lens-Profil laden
    profile = load_lens_profile(lens_profile_id)
    if not profile:
        return jsonify({"error": f"Lens-Profil '{lens_profile_id}' nicht gefunden"}), 404

    try:
        # Base64 zu Bild konvertieren
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Bild konnte nicht dekodiert werden"}), 400

        h, w = img.shape[:2]

        # Kameramatrix und Distortion aus Profil extrahieren
        fisheye_params = profile.get("fisheye_params", {})
        camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
        distortion_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64)

        # Distortion-Koeffizienten als (4,1) Spaltenvektor für OpenCV fisheye
        if distortion_coeffs.ndim == 1:
            distortion_coeffs = distortion_coeffs.reshape(4, 1)

        # Kameramatrix skalieren falls nötig
        profile_res = profile.get("resolution", {})
        profile_w = profile_res.get("width", w)
        profile_h = profile_res.get("height", h)

        if profile_w != w or profile_h != h:
            camera_matrix = scale_camera_matrix(
                camera_matrix,
                (profile_w, profile_h),
                (w, h)
            )

        # Bild entzerren
        undistorted = undistort_image(img, camera_matrix, distortion_coeffs, balance=balance)

        # Zurück zu Base64
        _, buffer = cv2.imencode('.jpg', undistorted, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "image_data": f"data:image/jpeg;base64,{result_base64}",
            "original_size": [w, h],
            "lens_profile": lens_profile_id
        })

    except Exception as e:
        import traceback
        print(f"Test undistort error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/estimate-distortion", methods=["POST"])
def api_estimate_distortion():
    """
    Schätzt Fisheye-Distortion-Parameter aus gezeichneten Hilfslinien.

    Request Body:
    {
        "lines": [
            {
                "points": [[x1, y1], [x2, y2], ...],
                "type": "horizontal" | "vertical" | "free"
            },
            ...
        ],
        "image_width": 3840,
        "image_height": 2160,
        "save_as": "mein_profil"  // optional
    }

    Response:
    {
        "camera_matrix": [[...], [...], [...]],
        "distortion_coeffs": [k1, k2, k3, k4],
        "estimated_k1": -0.123,
        "straightness_error": 2.5,
        "profile_id": "mein_profil"  // wenn save_as angegeben
    }
    """
    from src.processing.calibration import estimate_distortion_from_lines, save_estimated_profile

    data = request.get_json()
    if not data:
        return jsonify({"error": "Keine Daten empfangen"}), 400

    lines = data.get("lines", [])
    if len(lines) < 1:
        return jsonify({"error": "Mindestens eine Linie erforderlich"}), 400

    image_width = data.get("image_width", 3840)
    image_height = data.get("image_height", 2160)

    # Linien validieren und konvertieren
    valid_lines = []
    for line in lines:
        points = line.get("points", [])
        if len(points) < 3:
            continue
        valid_lines.append({
            "points": points,
            "type": line.get("type", "free")
        })

    if len(valid_lines) < 1:
        return jsonify({"error": "Mindestens eine Linie mit >= 3 Punkten erforderlich"}), 400

    try:
        # Schätzung durchführen
        result = estimate_distortion_from_lines(
            valid_lines,
            (image_width, image_height)
        )

        # Optional als Profil speichern
        save_as = data.get("save_as")
        if save_as:
            profile_path = save_estimated_profile(
                result,
                save_as,
                f"Geschätzt aus {len(valid_lines)} Hilfslinien"
            )
            result["profile_id"] = save_as
            result["profile_path"] = profile_path

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Schätzung fehlgeschlagen: {str(e)}"}), 500


# === Quick Calibration APIs ===

@calibration_bp.route("/api/videos/<video_id>/calibration/preview", methods=["POST"])
def api_calibration_preview(video_id):
    """
    Generiert Vorschau-Overlay basierend auf Punkt-Paaren.

    JSON Body:
        point_pairs: Liste von {image: {x,y}, field: {x,y}}
        use_undistorted: Boolean - verwende entzerrtes Bild

    Returns:
        success: Boolean
        overlay_path: Pfad zum generierten Overlay
    """
    import cv2
    import numpy as np

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    point_pairs = data.get("point_pairs", [])
    use_undistorted = data.get("use_undistorted", True)

    if len(point_pairs) < 4:
        return jsonify({"error": "Mindestens 4 Punkt-Paare erforderlich"}), 400

    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id

    # Bild laden
    if use_undistorted:
        img_path = calibration_dir / "undistorted_frame.jpg"
    else:
        img_path = calibration_dir / "calibration_frame.jpg"

    if not img_path.exists():
        return jsonify({"error": f"Bild nicht gefunden: {img_path.name}"}), 404

    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({"error": "Bild konnte nicht geladen werden"}), 500

        h, w = img.shape[:2]

        # Punkt-Arrays erstellen
        image_pts = np.array([[p["image"]["x"], p["image"]["y"]] for p in point_pairs], dtype=np.float32)
        field_pts = np.array([[p["field"]["x"], p["field"]["y"]] for p in point_pairs], dtype=np.float32)

        # Homography berechnen
        homography, status = cv2.findHomography(image_pts, field_pts, cv2.RANSAC, 5.0)

        if homography is None:
            return jsonify({"error": "Homography konnte nicht berechnet werden"}), 500

        # Inverse für Feld→Bild
        homography_inv = np.linalg.inv(homography)

        # Overlay zeichnen
        overlay = img.copy()

        # Spielfeld-Dimensionen
        FIELD_W, FIELD_H = 40, 20

        # Grid zeichnen (5m Abstand)
        for x in range(0, int(FIELD_W) + 1, 5):
            pts = []
            for y_step in np.linspace(0, FIELD_H, 50):
                field_pt = np.array([[[x, y_step]]], dtype=np.float32)
                img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
                pts.append(img_pt[0][0])
            pts = np.array(pts, dtype=np.int32)
            color = (0, 255, 255) if x == 20 else (0, 255, 0)  # Gelb für Mittellinie
            cv2.polylines(overlay, [pts], False, color, 2)

        for y in range(0, int(FIELD_H) + 1, 5):
            pts = []
            for x_step in np.linspace(0, FIELD_W, 100):
                field_pt = np.array([[[x_step, y]]], dtype=np.float32)
                img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
                pts.append(img_pt[0][0])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)

        # Spielfeldrand (rot)
        border_pts = []
        for pt in [(0, 0), (FIELD_W, 0), (FIELD_W, FIELD_H), (0, FIELD_H), (0, 0)]:
            field_pt = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
            border_pts.append(img_pt[0][0])
        border_pts = np.array(border_pts, dtype=np.int32)
        cv2.polylines(overlay, [border_pts], False, (0, 0, 255), 3)

        # Mittelkreis (cyan)
        circle_pts = []
        for angle in np.linspace(0, 2 * np.pi, 60):
            cx = 20 + 3 * np.cos(angle)
            cy = 10 + 3 * np.sin(angle)
            field_pt = np.array([[[cx, cy]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
            circle_pts.append(img_pt[0][0])
        circle_pts = np.array(circle_pts, dtype=np.int32)
        cv2.polylines(overlay, [circle_pts], True, (255, 255, 0), 2)

        # Torräume (magenta)
        for goal_x in [0, FIELD_W]:
            # Halbrund
            tor_pts = []
            for angle in np.linspace(-np.pi/2, np.pi/2, 30):
                sign = 1 if goal_x == 0 else -1
                cx = goal_x + sign * 2.85 * np.cos(angle)
                cy = 10 + 2.85 * np.sin(angle)
                if (goal_x == 0 and cx >= 0) or (goal_x == FIELD_W and cx <= FIELD_W):
                    field_pt = np.array([[[cx, cy]]], dtype=np.float32)
                    img_pt = cv2.perspectiveTransform(field_pt, homography_inv)
                    tor_pts.append(img_pt[0][0])
            if tor_pts:
                tor_pts = np.array(tor_pts, dtype=np.int32)
                cv2.polylines(overlay, [tor_pts], False, (255, 0, 255), 2)

        # Referenzpunkte einzeichnen (orange)
        for i, pair in enumerate(point_pairs):
            x, y = int(pair["image"]["x"]), int(pair["image"]["y"])
            cv2.circle(overlay, (x, y), 8, (0, 165, 255), -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(overlay, str(i+1), (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Speichern
        output_path = calibration_dir / "test_overlay.jpg"
        cv2.imwrite(str(output_path), overlay)

        return jsonify({
            "success": True,
            "overlay_path": str(output_path)
        })

    except Exception as e:
        import traceback
        print(f"Preview error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# === Work-in-Progress (WIP) Speicherung ===

@calibration_bp.route("/api/videos/<video_id>/calibration/wip", methods=["GET"])
def api_get_calibration_wip(video_id):
    """
    Lädt Work-in-Progress Kalibrierungsdaten.
    """
    import json

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    wip_path = PROJECT_ROOT / "data" / "calibration" / video_id / "wip_calibration.json"

    if not wip_path.exists():
        return jsonify({"exists": False})

    try:
        with open(wip_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"exists": True, "data": data})
    except Exception as e:
        return jsonify({"exists": False, "error": str(e)})


@calibration_bp.route("/api/videos/<video_id>/calibration/wip", methods=["POST"])
def api_save_calibration_wip(video_id):
    """
    Speichert Work-in-Progress Kalibrierungsdaten.

    JSON Body:
        point_pairs: Liste von Punkt-Paaren
        undistort_params: Entzerrungs-Parameter
        selected_screenshots: Ausgewählte Screenshot-Indizes
        active_screenshot: Aktueller Screenshot-Index
        current_step: Aktueller Schritt
    """
    import json
    from datetime import datetime

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}

    wip_data = {
        "video_id": video_id,
        "updated_at": datetime.now().isoformat(),
        "point_pairs": data.get("point_pairs", []),
        "undistort_params": data.get("undistort_params", {}),
        "selected_screenshots": data.get("selected_screenshots", []),
        "active_screenshot": data.get("active_screenshot"),
        "current_step": data.get("current_step", 1)
    }

    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id
    calibration_dir.mkdir(parents=True, exist_ok=True)

    wip_path = calibration_dir / "wip_calibration.json"

    try:
        with open(wip_path, "w", encoding="utf-8") as f:
            json.dump(wip_data, f, indent=2, ensure_ascii=False)
        return jsonify({"success": True, "path": str(wip_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/calibration/wip", methods=["DELETE"])
def api_delete_calibration_wip(video_id):
    """
    Löscht Work-in-Progress Kalibrierungsdaten (nach erfolgreichem Speichern).
    """
    wip_path = PROJECT_ROOT / "data" / "calibration" / video_id / "wip_calibration.json"

    if wip_path.exists():
        wip_path.unlink()

    return jsonify({"success": True})


@calibration_bp.route("/api/videos/<video_id>/calibration/save", methods=["POST"])
def api_calibration_save(video_id):
    """
    Speichert die Kalibrierung.

    JSON Body:
        point_pairs: Liste von {image: {x,y}, field: {x,y}}
        use_undistorted: Boolean
        lens_profile_id: Optional
        undistort_params: Optional - Custom Entzerrungs-Parameter
        name: Optional - Name der Kalibrierung

    Returns:
        success: Boolean
        calibration_id: ID der gespeicherten Kalibrierung
    """
    import cv2
    import numpy as np
    import json

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    point_pairs = data.get("point_pairs", [])
    use_undistorted = data.get("use_undistorted", True)
    lens_profile_id = data.get("lens_profile_id")
    undistort_params = data.get("undistort_params", {})
    name = data.get("name", "Kalibrierung")
    calibration_id = data.get("calibration_id")  # Optional: ID für Update

    if len(point_pairs) < 4:
        return jsonify({"error": "Mindestens 4 Punkt-Paare erforderlich"}), 400

    try:
        # Punkt-Arrays erstellen
        image_pts = np.array([[p["image"]["x"], p["image"]["y"]] for p in point_pairs], dtype=np.float32)
        field_pts = np.array([[p["field"]["x"], p["field"]["y"]] for p in point_pairs], dtype=np.float32)

        # Homography berechnen
        homography, status = cv2.findHomography(image_pts, field_pts, cv2.RANSAC, 5.0)

        if homography is None:
            return jsonify({"error": "Homography konnte nicht berechnet werden"}), 500

        # Kalibrierungsdaten zusammenstellen
        calibration_data = {
            "name": name,
            "point_pairs": point_pairs,
            "homography_matrix": homography.tolist(),
            "use_undistorted": use_undistorted,
            "lens_profile_id": lens_profile_id,
            "image_points": image_pts.tolist(),
            "field_points": field_pts.tolist()
        }

        # Undistort-Parameter hinzufügen (für Reproduzierbarkeit)
        if undistort_params:
            calibration_data["undistort_params"] = undistort_params

        # Bestehende Kalibrierung aktualisieren oder neue erstellen
        if calibration_id:
            # Update: Bestehende Kalibrierung aktualisieren
            calibration = db.session.get(Calibration, calibration_id)
            if not calibration or calibration.video_id != video.id:
                return jsonify({"error": "Kalibrierung nicht gefunden"}), 404

            calibration.calibration_data = calibration_data
            calibration.name = name
        else:
            # Neue Kalibrierung: Andere deaktivieren
            db.session.query(Calibration).filter_by(
                video_id=video.id,
                is_active=True
            ).update({"is_active": False})

            calibration = Calibration(
                video_id=video.id,
                is_active=True,
                name=name,
                calibration_data=calibration_data
            )
            db.session.add(calibration)

        db.session.commit()

        # JSON-Datei speichern für einfachen Zugriff/Export
        calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id
        calibration_dir.mkdir(parents=True, exist_ok=True)

        json_path = calibration_dir / f"calibration_{calibration.id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": str(calibration.id),
                "video_id": video_id,
                "created_at": calibration.created_at.isoformat() if calibration.created_at else None,
                **calibration_data
            }, f, indent=2, ensure_ascii=False)

        return jsonify({
            "success": True,
            "calibration_id": str(calibration.id),
            "json_path": str(json_path)
        })

    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"Save error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# === Distorted Overlay API ===

@calibration_bp.route("/api/videos/<video_id>/calibration/distorted-overlay", methods=["POST"])
def api_distorted_overlay(video_id):
    """
    Generiert ein Overlay auf dem VERZERRTEN (Original) Bild.

    Der Workflow:
    1. Punkt-Paare definieren die Homographie (auf dem entzerrten Bild)
    2. Spielfeld-Linien werden in viele kurze Segmente unterteilt
    3. Jeder Punkt wird mit der inversen Homographie auf Bildkoordinaten transformiert
    4. Dann mit der Lens-Distortion wieder verzerrt
    5. Das Ergebnis sind gekrümmte Linien auf dem Original-Bild

    JSON Body:
        point_pairs: Liste von {image: {x,y}, field: {x,y}}
        lens_profile_id: ID des Lens-Profils ODER "custom"
        undistort_params: (optional) Custom-Parameter wenn lens_profile_id="custom"
            - base_profile_id: Basis-Profil für Kameramatrix
            - k1, k2, k3, k4: Distortion-Koeffizienten
            - zoom_out, rotation: Entzerrungs-Parameter

    Returns:
        success: Boolean
        overlay_path: Pfad zum generierten Overlay
    """
    import cv2
    import numpy as np
    from src.processing.calibration import load_lens_profile, scale_camera_matrix, distort_points

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    point_pairs = data.get("point_pairs", [])
    lens_profile_id = data.get("lens_profile_id")
    screenshot_filename = data.get("screenshot_filename")
    undistort_params = data.get("undistort_params", {})

    if len(point_pairs) < 4:
        return jsonify({"error": "Mindestens 4 Punkt-Paare erforderlich"}), 400

    # Prüfen ob Custom-Modus oder Profil
    is_custom_mode = lens_profile_id == "custom"

    if not lens_profile_id and not is_custom_mode:
        return jsonify({"error": "lens_profile_id erforderlich für Rücktransformation"}), 400

    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id

    # Original (verzerrtes) Bild laden
    original_path = None

    # Zuerst: Wenn ein spezifischer Screenshot angegeben wurde
    if screenshot_filename:
        candidate_path = calibration_dir / "screenshots" / screenshot_filename
        if candidate_path.exists():
            original_path = candidate_path

    # Fallback auf andere Screenshots
    if not original_path:
        for candidate in ["screenshots/frame_0000.jpg", "calibration_frame.jpg"]:
            candidate_path = calibration_dir / candidate
            if candidate_path.exists():
                original_path = candidate_path
                break

    # Letzter Fallback: Erster verfügbarer Screenshot
    if not original_path:
        screenshot_files = list((calibration_dir / "screenshots").glob("frame_*.jpg"))
        if screenshot_files:
            original_path = sorted(screenshot_files)[0]

    if not original_path or not original_path.exists():
        return jsonify({"error": "Original-Screenshot nicht gefunden"}), 404

    # Lens-Profil/Parameter laden
    profile = None
    if is_custom_mode:
        # Custom-Modus: Lade Basis-Profil für Kameramatrix, verwende Custom-Koeffizienten
        base_profile_id = undistort_params.get("base_profile_id")
        if base_profile_id:
            profile = load_lens_profile(base_profile_id)
        if not profile:
            # Fallback: Standard GoPro Profil
            profile = load_lens_profile("hero10_standard_wide_4k")
        if not profile:
            return jsonify({"error": "Kein Basis-Profil für Custom-Modus gefunden"}), 404

        # Custom k1-k4 Koeffizienten überschreiben
        k1 = undistort_params.get("k1", 0.15)
        k2 = undistort_params.get("k2", -0.15)
        k3 = undistort_params.get("k3", 0.1)
        k4 = undistort_params.get("k4", -0.04)
        profile["fisheye_params"]["distortion_coeffs"] = [k1, k2, k3, k4]
    else:
        profile = load_lens_profile(lens_profile_id)
        if not profile:
            return jsonify({"error": f"Lens-Profil '{lens_profile_id}' nicht gefunden"}), 404

    try:
        # Original-Bild laden
        img = cv2.imread(str(original_path))
        if img is None:
            return jsonify({"error": "Bild konnte nicht geladen werden"}), 500

        h, w = img.shape[:2]

        # Kameramatrix und Distortion aus Profil
        fisheye_params = profile.get("fisheye_params", {})
        camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
        distortion_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64)

        if distortion_coeffs.ndim == 1:
            distortion_coeffs = distortion_coeffs.reshape(4, 1)

        # Skalieren falls nötig
        profile_res = profile.get("resolution", {})
        profile_w = profile_res.get("width", w)
        profile_h = profile_res.get("height", h)

        if profile_w != w or profile_h != h:
            camera_matrix = scale_camera_matrix(
                camera_matrix,
                (profile_w, profile_h),
                (w, h)
            )

        # Homography berechnen (auf entzerrtem Bild-Koordinaten)
        image_pts = np.array([[p["image"]["x"], p["image"]["y"]] for p in point_pairs], dtype=np.float32)
        field_pts = np.array([[p["field"]["x"], p["field"]["y"]] for p in point_pairs], dtype=np.float32)

        homography, _ = cv2.findHomography(image_pts, field_pts, cv2.RANSAC, 5.0)
        if homography is None:
            return jsonify({"error": "Homography konnte nicht berechnet werden"}), 500

        homography_inv = np.linalg.inv(homography)

        # Spielfeld-Dimensionen
        FIELD_W, FIELD_H = 40, 20
        SEGMENT_LENGTH = 0.5  # Meter pro Segment (kurz für glatte Kurven)

        overlay = img.copy()

        # Zoom und Rotation aus undistort_params holen (IMMER, nicht nur Custom-Modus)
        # Priorität: 1. Request undistort_params, 2. Profil undistort_params, 3. Defaults
        zoom_out = 1.2
        rotation = 0.0

        # Aus Request-undistort_params
        if undistort_params:
            zoom_out = undistort_params.get("zoom_out", zoom_out)
            rotation = undistort_params.get("rotation", rotation)

        # Fallback: Aus Profil (falls nicht im Request)
        if profile.get("undistort_params"):
            if "zoom_out" not in undistort_params or undistort_params.get("zoom_out") is None:
                zoom_out = profile["undistort_params"].get("zoom_out", zoom_out)
            if "rotation" not in undistort_params or undistort_params.get("rotation") is None:
                rotation = profile["undistort_params"].get("rotation", rotation)

        print(f"Distorted overlay: zoom_out={zoom_out}, rotation={rotation}")

        # Die "gezoomed" Kameramatrix, die für die Entzerrung verwendet wurde
        zoomed_K = camera_matrix.copy()
        zoomed_K[0, 0] /= zoom_out
        zoomed_K[1, 1] /= zoom_out

        def transform_and_distort_line(field_points):
            """
            Transformiert Feldpunkte zu Bildpunkten und wendet Distortion an.

            Der Workflow (WICHTIG - Reihenfolge ist kritisch!):
            Bei der Entzerrung passierte: Original → Undistort → Rotate
            Bei der Rücktransformation muss es sein: Rotate⁻¹ → Distort

            1. Feldpunkte → entzerrte Bildpunkte (via inverse Homographie)
               Diese Punkte sind im rotierten, entzerrten Koordinatensystem
            2. Rotation rückgängig machen (ZUERST!)
               Jetzt sind die Punkte im nicht-rotierten, entzerrten System
            3. Von gezoomter zu originaler Kameramatrix konvertieren
            4. Fisheye-Verzerrung anwenden
            """
            # Feld → Bild (entzerrt + rotiert, mit Zoom)
            field_pts_arr = np.array([[[p[0], p[1]]] for p in field_points], dtype=np.float32)
            undistorted_pts = cv2.perspectiveTransform(field_pts_arr, homography_inv)
            undistorted_pts_2d = undistorted_pts.reshape(-1, 2)

            # SCHRITT 1: Rotation rückgängig machen (ZUERST, bevor Fisheye!)
            # Das Bild wurde nach der Entzerrung rotiert, also müssen wir
            # die Punkte zurück-rotieren bevor wir die Verzerrung anwenden
            #
            # WICHTIG: Wenn das BILD um +R Grad rotiert wurde, dann erscheinen
            # die PUNKTE an anderen Stellen. Um sie zurückzutransformieren,
            # müssen wir die PUNKTE um +R Grad rotieren (nicht -R!),
            # weil wir die inverse Bildtransformation auf Punkte anwenden.
            if abs(rotation) > 0.01:
                center = np.array([w / 2, h / 2])
                # Positive Rotation auf Punkte = inverse der Bildrotation
                angle_rad = np.radians(rotation)  # POSITIV, nicht negativ!
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rotated_pts = np.zeros_like(undistorted_pts_2d)
                for i, pt in enumerate(undistorted_pts_2d):
                    translated = pt - center
                    rotated_pts[i, 0] = translated[0] * cos_a - translated[1] * sin_a + center[0]
                    rotated_pts[i, 1] = translated[0] * sin_a + translated[1] * cos_a + center[1]
                undistorted_pts_2d = rotated_pts
                print(f"Applied rotation correction: {rotation}° on {len(rotated_pts)} points")

            # SCHRITT 2: Von gezoomter Kameramatrix zu normalisierten Koordinaten
            fx_zoom, fy_zoom = zoomed_K[0, 0], zoomed_K[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            # Normalisierte Koordinaten (3D-Richtungen auf Z=1 Ebene)
            normalized = np.zeros_like(undistorted_pts_2d)
            normalized[:, 0] = (undistorted_pts_2d[:, 0] - cx) / fx_zoom
            normalized[:, 1] = (undistorted_pts_2d[:, 1] - cy) / fy_zoom

            # SCHRITT 3: Fisheye-Verzerrung anwenden
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
                distortion_coeffs
            )

            return distorted.reshape(-1, 2)

        def draw_segmented_line(start, end, color, thickness, num_segments=None):
            """Zeichnet eine Linie als viele kurze Segmente."""
            dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            if num_segments is None:
                num_segments = max(int(dist / SEGMENT_LENGTH), 2)

            field_points = []
            for i in range(num_segments + 1):
                t = i / num_segments
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                field_points.append((x, y))

            distorted = transform_and_distort_line(field_points)
            pts = np.array(distorted, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)

        def draw_segmented_circle(center, radius, color, thickness, num_points=60):
            """Zeichnet einen Kreis als viele kurze Segmente."""
            field_points = []
            for i in range(num_points + 1):
                angle = 2 * np.pi * i / num_points
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                field_points.append((x, y))

            distorted = transform_and_distort_line(field_points)
            pts = np.array(distorted, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, color, thickness, cv2.LINE_AA)

        def draw_segmented_arc(center, radius, start_angle, end_angle, color, thickness, num_points=30):
            """Zeichnet einen Bogen als Segmente."""
            field_points = []
            for i in range(num_points + 1):
                t = i / num_points
                angle = start_angle + t * (end_angle - start_angle)
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                # Nur Punkte innerhalb des Spielfelds
                if 0 <= x <= FIELD_W and 0 <= y <= FIELD_H:
                    field_points.append((x, y))

            if len(field_points) >= 2:
                distorted = transform_and_distort_line(field_points)
                pts = np.array(distorted, dtype=np.int32)
                cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)

        # === SPIELFELD ZEICHNEN ===

        # Grid (5m Abstand) - Grün
        for x in range(0, int(FIELD_W) + 1, 5):
            color = (0, 255, 255) if x == 20 else (0, 255, 0)  # Gelb für Mittellinie
            thickness = 3 if x == 20 else 2
            draw_segmented_line((x, 0), (x, FIELD_H), color, thickness)

        for y in range(0, int(FIELD_H) + 1, 5):
            draw_segmented_line((0, y), (FIELD_W, y), (0, 255, 0), 2)

        # Spielfeldrand - Rot, dick
        draw_segmented_line((0, 0), (FIELD_W, 0), (0, 0, 255), 4)
        draw_segmented_line((FIELD_W, 0), (FIELD_W, FIELD_H), (0, 0, 255), 4)
        draw_segmented_line((FIELD_W, FIELD_H), (0, FIELD_H), (0, 0, 255), 4)
        draw_segmented_line((0, FIELD_H), (0, 0), (0, 0, 255), 4)

        # Mittelkreis (3m Radius) - Cyan
        draw_segmented_circle((20, 10), 3, (255, 255, 0), 2)

        # Torräume (2.85m Halbkreis) - Magenta
        draw_segmented_arc((0, 10), 2.85, -np.pi/2, np.pi/2, (255, 0, 255), 2)
        draw_segmented_arc((40, 10), 2.85, np.pi/2, 3*np.pi/2, (255, 0, 255), 2)

        # Referenzpunkte einzeichnen (orange)
        # Nutzen dieselbe transform_and_distort_line Logik für konsistente Transformation
        for i, pair in enumerate(point_pairs):
            # Einzelpunkt als "Linie" mit einem Punkt transformieren
            pt = [(pair["field"]["x"], pair["field"]["y"])]
            dist_pt = transform_and_distort_line(pt)
            x, y = int(dist_pt[0][0]), int(dist_pt[0][1])

            cv2.circle(overlay, (x, y), 10, (0, 165, 255), -1)
            cv2.circle(overlay, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(overlay, str(i+1), (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Speichern
        output_path = calibration_dir / "distorted_overlay.jpg"
        cv2.imwrite(str(output_path), overlay)

        return jsonify({
            "success": True,
            "overlay_path": str(output_path)
        })

    except Exception as e:
        import traceback
        print(f"Distorted overlay error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# === Static Files ===

@calibration_bp.route("/api/videos/<video_id>/custom-undistort", methods=["POST"])
def api_custom_undistort(video_id):
    """
    Entzerrt einen Screenshot mit benutzerdefinierten k1-k4 Parametern.
    Basis-Parameter werden von einem existierenden Lens-Profil übernommen.

    JSON Body:
        base_profile_id: ID des Basis-Lens-Profils (für Kameramatrix)
        k1, k2, k3, k4: Distortion-Koeffizienten (optional, überschreiben Basis-Profil)
        zoom_out: Zoom-Faktor (default: 1.2)
        screenshot_path: Pfad zum Screenshot (optional)
        draw_guidelines: Boolean - horizontale Hilfslinien zeichnen (default: true)
        guideline_count: Anzahl Hilfslinien (default: 10)

    Returns:
        undistorted_path: Pfad zum entzerrten Bild mit Hilfslinien
    """
    import cv2
    import numpy as np
    from src.processing.calibration import load_lens_profile, scale_camera_matrix

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    base_profile_id = data.get("base_profile_id")
    zoom_out = data.get("zoom_out", 1.2)
    rotation = data.get("rotation", 0.0)  # Rotation in Grad
    draw_guidelines = data.get("draw_guidelines", True)
    guideline_count = data.get("guideline_count", 10)

    # Distortion-Koeffizienten
    k1 = data.get("k1")
    k2 = data.get("k2")
    k3 = data.get("k3")
    k4 = data.get("k4")

    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id

    # Screenshot-Pfad bestimmen
    screenshot_path = data.get("screenshot_path")
    if not screenshot_path:
        for candidate in ["calibration_frame.jpg", "screenshots/frame_0000.jpg"]:
            candidate_path = calibration_dir / candidate
            if candidate_path.exists():
                screenshot_path = str(candidate_path)
                break

    if not screenshot_path or not os.path.exists(screenshot_path):
        return jsonify({"error": "Screenshot nicht gefunden"}), 404

    try:
        # Bild laden
        img = cv2.imread(screenshot_path)
        if img is None:
            return jsonify({"error": "Screenshot konnte nicht geladen werden"}), 500

        h, w = img.shape[:2]

        # Kameramatrix aus Basis-Profil oder Standard für GoPro Hero 10
        if base_profile_id:
            profile = load_lens_profile(base_profile_id)
            if not profile:
                return jsonify({"error": f"Basis-Profil '{base_profile_id}' nicht gefunden"}), 404

            fisheye_params = profile.get("fisheye_params", {})
            camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
            base_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64).flatten()

            # Skalieren falls nötig
            profile_res = profile.get("resolution", {})
            profile_w = profile_res.get("width", w)
            profile_h = profile_res.get("height", h)

            if profile_w != w or profile_h != h:
                camera_matrix = scale_camera_matrix(camera_matrix, (profile_w, profile_h), (w, h))
        else:
            # Standard GoPro Hero 10 Kameramatrix (4K)
            fx = w * 0.415  # Typischer Wert für GoPro
            fy = fx
            cx = w / 2
            cy = h / 2
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            base_coeffs = np.array([0.15, -0.15, 0.1, -0.04])  # Typische GoPro-Werte

        # Custom-Koeffizienten anwenden
        distortion_coeffs = np.array([
            k1 if k1 is not None else base_coeffs[0],
            k2 if k2 is not None else base_coeffs[1],
            k3 if k3 is not None else base_coeffs[2],
            k4 if k4 is not None else base_coeffs[3]
        ], dtype=np.float64).reshape(4, 1)

        # Neue Kameramatrix für Zoom-Out
        new_K = camera_matrix.copy()
        new_K[0, 0] /= zoom_out
        new_K[1, 1] /= zoom_out

        # Undistortion Maps berechnen
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, distortion_coeffs,
            np.eye(3), new_K,
            (w, h), cv2.CV_16SC2
        )

        # Entzerren
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Rotation anwenden (falls nicht 0)
        if abs(rotation) > 0.01:
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            undistorted = cv2.warpAffine(undistorted, rotation_matrix, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))

        # Hilfslinien zeichnen
        if draw_guidelines:
            # Horizontale Linien
            for i in range(1, guideline_count):
                y_pos = int(h * i / guideline_count)
                cv2.line(undistorted, (0, y_pos), (w, y_pos), (0, 255, 255), 1, cv2.LINE_AA)

            # Vertikale Linien (weniger, hauptsächlich Mitte und Drittel)
            for x_ratio in [0.25, 0.5, 0.75]:
                x_pos = int(w * x_ratio)
                color = (0, 255, 0) if x_ratio == 0.5 else (0, 200, 200)
                thickness = 2 if x_ratio == 0.5 else 1
                cv2.line(undistorted, (x_pos, 0), (x_pos, h), color, thickness, cv2.LINE_AA)

            # Info-Text
            info_text = f"k1={distortion_coeffs[0][0]:.4f}  k2={distortion_coeffs[1][0]:.4f}  k3={distortion_coeffs[2][0]:.4f}  k4={distortion_coeffs[3][0]:.4f}  zoom={zoom_out:.2f}  rot={rotation:.1f}"
            cv2.putText(undistorted, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(undistorted, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Speichern
        output_path = calibration_dir / "custom_undistorted.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), undistorted)

        return jsonify({
            "success": True,
            "undistorted_path": str(output_path),
            "params": {
                "k1": float(distortion_coeffs[0][0]),
                "k2": float(distortion_coeffs[1][0]),
                "k3": float(distortion_coeffs[2][0]),
                "k4": float(distortion_coeffs[3][0]),
                "zoom_out": zoom_out,
                "rotation": rotation
            }
        })

    except Exception as e:
        import traceback
        print(f"Custom undistort error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/videos/<video_id>/custom-undistort-with-trapez", methods=["POST"])
def api_custom_undistort_with_trapez(video_id):
    """
    Entzerrt einen Screenshot und zeichnet ein Trapez mit Fluchtlinien.

    JSON Body:
        ... (wie custom-undistort)
        trapez_points: Liste von 4 Punkten [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                      Reihenfolge: oben-links, oben-rechts, unten-rechts, unten-links
    """
    import cv2
    import numpy as np
    from src.processing.calibration import load_lens_profile, scale_camera_matrix

    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    data = request.get_json() or {}
    base_profile_id = data.get("base_profile_id")
    zoom_out = data.get("zoom_out", 1.2)
    rotation = data.get("rotation", 0.0)  # Rotation in Grad
    trapez_points = data.get("trapez_points", [])
    draw_guidelines = data.get("draw_guidelines", True)

    # Distortion-Koeffizienten
    k1 = data.get("k1")
    k2 = data.get("k2")
    k3 = data.get("k3")
    k4 = data.get("k4")

    calibration_dir = PROJECT_ROOT / "data" / "calibration" / video_id

    # Screenshot-Pfad bestimmen
    screenshot_path = data.get("screenshot_path")
    if not screenshot_path:
        for candidate in ["calibration_frame.jpg", "screenshots/frame_0000.jpg"]:
            candidate_path = calibration_dir / candidate
            if candidate_path.exists():
                screenshot_path = str(candidate_path)
                break

    if not screenshot_path or not os.path.exists(screenshot_path):
        return jsonify({"error": "Screenshot nicht gefunden"}), 404

    try:
        # Bild laden
        img = cv2.imread(screenshot_path)
        if img is None:
            return jsonify({"error": "Screenshot konnte nicht geladen werden"}), 500

        h, w = img.shape[:2]

        # Kameramatrix und Distortion
        if base_profile_id:
            profile = load_lens_profile(base_profile_id)
            if profile:
                fisheye_params = profile.get("fisheye_params", {})
                camera_matrix = np.array(fisheye_params.get("camera_matrix"), dtype=np.float64)
                base_coeffs = np.array(fisheye_params.get("distortion_coeffs"), dtype=np.float64).flatten()

                profile_res = profile.get("resolution", {})
                profile_w = profile_res.get("width", w)
                profile_h = profile_res.get("height", h)

                if profile_w != w or profile_h != h:
                    camera_matrix = scale_camera_matrix(camera_matrix, (profile_w, profile_h), (w, h))
            else:
                return jsonify({"error": f"Basis-Profil '{base_profile_id}' nicht gefunden"}), 404
        else:
            fx = w * 0.415
            fy = fx
            cx = w / 2
            cy = h / 2
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            base_coeffs = np.array([0.15, -0.15, 0.1, -0.04])

        distortion_coeffs = np.array([
            k1 if k1 is not None else base_coeffs[0],
            k2 if k2 is not None else base_coeffs[1],
            k3 if k3 is not None else base_coeffs[2],
            k4 if k4 is not None else base_coeffs[3]
        ], dtype=np.float64).reshape(4, 1)

        # Neue Kameramatrix für Zoom-Out
        new_K = camera_matrix.copy()
        new_K[0, 0] /= zoom_out
        new_K[1, 1] /= zoom_out

        # Undistortion Maps berechnen
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, distortion_coeffs,
            np.eye(3), new_K,
            (w, h), cv2.CV_16SC2
        )

        # Entzerren
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Rotation anwenden (falls nicht 0)
        if abs(rotation) > 0.01:
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            undistorted = cv2.warpAffine(undistorted, rotation_matrix, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))

        # Hilfslinien
        if draw_guidelines:
            for i in range(1, 10):
                y_pos = int(h * i / 10)
                cv2.line(undistorted, (0, y_pos), (w, y_pos), (0, 255, 255), 1, cv2.LINE_AA)

            for x_ratio in [0.25, 0.5, 0.75]:
                x_pos = int(w * x_ratio)
                color = (0, 255, 0) if x_ratio == 0.5 else (0, 200, 200)
                cv2.line(undistorted, (x_pos, 0), (x_pos, h), color, 1, cv2.LINE_AA)

        # Trapez und Fluchtlinien zeichnen
        if len(trapez_points) == 4:
            pts = np.array(trapez_points, dtype=np.int32)

            # Trapez zeichnen (magenta)
            cv2.polylines(undistorted, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)

            # Punkte markieren
            for i, pt in enumerate(trapez_points):
                cv2.circle(undistorted, (int(pt[0]), int(pt[1])), 6, (255, 0, 255), -1)
                cv2.putText(undistorted, str(i+1), (int(pt[0])+8, int(pt[1])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Fluchtpunkt berechnen (Schnittpunkt der verlängerten Seiten)
            # Linke Seite: Punkt 0 → Punkt 3
            # Rechte Seite: Punkt 1 → Punkt 2
            p0, p1, p2, p3 = [np.array(p) for p in trapez_points]

            # Verlängerte Linien nach oben (Fluchtpunkt)
            def line_intersection(p1, p2, p3, p4):
                """Berechnet Schnittpunkt zweier Linien."""
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                x4, y4 = p4

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None  # Parallel

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                return (x, y)

            # Fluchtpunkt der vertikalen Seiten
            vanishing_point = line_intersection(p0, p3, p1, p2)

            if vanishing_point:
                vx, vy = vanishing_point

                # Fluchtlinien zeichnen (cyan, gestrichelt simuliert durch kurze Segmente)
                # Von oberen Ecken zum Fluchtpunkt
                cv2.line(undistorted, (int(p0[0]), int(p0[1])), (int(vx), int(vy)), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.line(undistorted, (int(p1[0]), int(p1[1])), (int(vx), int(vy)), (255, 255, 0), 1, cv2.LINE_AA)

                # Fluchtpunkt markieren (wenn im Bild)
                if 0 <= vx <= w and 0 <= vy <= h:
                    cv2.circle(undistorted, (int(vx), int(vy)), 10, (0, 255, 255), 2)
                    cv2.putText(undistorted, "VP", (int(vx)+12, int(vy)+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Info-Text
        info_text = f"k1={distortion_coeffs[0][0]:.4f}  k2={distortion_coeffs[1][0]:.4f}  k3={distortion_coeffs[2][0]:.4f}  k4={distortion_coeffs[3][0]:.4f}  rot={rotation:.1f}"
        cv2.putText(undistorted, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(undistorted, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Speichern
        output_path = calibration_dir / "custom_undistorted.jpg"
        cv2.imwrite(str(output_path), undistorted)

        return jsonify({
            "success": True,
            "undistorted_path": str(output_path),
            "params": {
                "k1": float(distortion_coeffs[0][0]),
                "k2": float(distortion_coeffs[1][0]),
                "k3": float(distortion_coeffs[2][0]),
                "k4": float(distortion_coeffs[3][0]),
                "zoom_out": zoom_out,
                "rotation": rotation
            },
            "vanishing_point": vanishing_point if len(trapez_points) == 4 else None
        })

    except Exception as e:
        import traceback
        print(f"Custom undistort with trapez error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@calibration_bp.route("/api/lens-profiles/save-custom", methods=["POST"])
def api_save_custom_lens_profile():
    """
    Speichert ein Custom-Lens-Profil mit benutzerdefinierten k1-k4 Parametern.

    JSON Body:
        name: Name des Profils
        base_profile_id: ID des Basis-Profils (für Kameramatrix)
        k1, k2, k3, k4: Distortion-Koeffizienten
        resolution: {width, height}
    """
    import json
    from datetime import datetime
    from src.processing.calibration import load_lens_profile

    data = request.get_json() or {}
    name = data.get("name")
    base_profile_id = data.get("base_profile_id")

    if not name:
        return jsonify({"error": "Name erforderlich"}), 400

    # Basis-Profil laden
    if base_profile_id:
        base_profile = load_lens_profile(base_profile_id)
        if not base_profile:
            return jsonify({"error": f"Basis-Profil '{base_profile_id}' nicht gefunden"}), 404

        camera_matrix = base_profile.get("fisheye_params", {}).get("camera_matrix")
        resolution = base_profile.get("resolution", {"width": 3840, "height": 2160})
    else:
        # Standard GoPro 10 Matrix
        resolution = data.get("resolution", {"width": 3840, "height": 2160})
        w, h = resolution["width"], resolution["height"]
        fx = w * 0.415
        camera_matrix = [[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]]

    # Custom Distortion-Koeffizienten
    k1 = data.get("k1", 0.15)
    k2 = data.get("k2", -0.15)
    k3 = data.get("k3", 0.1)
    k4 = data.get("k4", -0.04)

    # Zoom und Rotation
    zoom_out = data.get("zoom_out", 1.2)
    rotation = data.get("rotation", 0.0)

    profile = {
        "name": name,
        "camera_name": "Custom (basierend auf " + (base_profile_id or "Standard GoPro 10") + ")",
        "lens_type": "fisheye",
        "resolution": resolution,
        "fisheye_params": {
            "camera_matrix": camera_matrix,
            "distortion_coeffs": [k1, k2, k3, k4]
        },
        "undistort_params": {
            "zoom_out": zoom_out,
            "rotation": rotation
        },
        "base_profile_id": base_profile_id,
        "created_at": datetime.utcnow().isoformat(),
        "source": "Manuelle Anpassung"
    }

    # Speichern
    profiles_dir = PROJECT_ROOT / "data" / "lens_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
    profile_path = profiles_dir / f"custom_{safe_name}.json"

    counter = 1
    while profile_path.exists():
        profile_path = profiles_dir / f"custom_{safe_name}_{counter}.json"
        counter += 1

    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)

    return jsonify({
        "success": True,
        "id": profile_path.stem,
        "path": str(profile_path)
    })


@calibration_bp.route("/screenshot/<video_id>/<path:filename>")
def serve_screenshot(video_id, filename):
    """Liefert Screenshot-Dateien aus (inkl. Unterordner wie screenshots/)."""
    file_path = PROJECT_ROOT / "data" / "calibration" / video_id / filename
    if not file_path.exists():
        abort(404)

    return send_file(file_path, mimetype="image/jpeg")
