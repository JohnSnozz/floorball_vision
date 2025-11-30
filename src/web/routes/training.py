"""
Training Routes

Endpoints für YOLO Training.
"""
import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, abort
from src.web.extensions import db
from src.db.models import LabelingProject, TrainingRun, ActiveModel
from src.processing.tasks import start_training

training_bp = Blueprint("training", __name__, url_prefix="/training")


def load_yolo_config():
    """Lädt die YOLO-Modell-Konfiguration."""
    config_path = Path("configs/yolo_models.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"models": {}, "recommendations": {}, "training_defaults": {}, "augmentation_presets": {}}


def get_available_exports():
    """Findet alle verfügbaren Export-Ordner."""
    exports_dir = Path("data/labeling/exports")
    exports = []

    if exports_dir.exists():
        for export_path in exports_dir.iterdir():
            if export_path.is_dir():
                data_yaml = export_path / "data.yaml"
                if data_yaml.exists():
                    # Zähle Bilder und Labels
                    images_dir = export_path / "images"
                    labels_dir = export_path / "labels"

                    image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
                    label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

                    exports.append({
                        "name": export_path.name,
                        "path": str(export_path),
                        "data_yaml": str(data_yaml),
                        "images": image_count,
                        "labels": label_count,
                    })

    return exports


# === Seiten ===

@training_bp.route("/")
def index():
    """Training Übersicht."""
    runs = db.session.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
    active = db.session.query(ActiveModel).first()
    config = load_yolo_config()
    exports = get_available_exports()

    return render_template(
        "training/index.html",
        runs=runs,
        active_model=active,
        yolo_models=config.get("models", {}),
        recommendations=config.get("recommendations", {}),
        training_defaults=config.get("training_defaults", {}),
        augmentation_presets=config.get("augmentation_presets", {}),
        available_exports=exports
    )


@training_bp.route("/<run_id>")
def detail(run_id):
    """Training Detail-Seite."""
    run = db.session.get(TrainingRun, run_id)
    if not run:
        abort(404)

    return render_template(
        "training/detail.html",
        run=run
    )


# === API Endpoints ===

@training_bp.route("/api/start", methods=["POST"])
def api_start_training():
    """
    Startet einen neuen Training-Run.

    JSON Body:
        export_path: Pfad zum Export-Ordner mit data.yaml
        base_model: Basis-Modell (z.B. yolov8n.pt)
        epochs: Anzahl Epochs (default: 100)
        batch_size: Batch-Grösse (default: 16)
        image_size: Bildgrösse (default: 1080)
        train_split: Anteil Training (default: 0.8)
        val_split: Anteil Validation (default: 0.2)
        augmentation: Augmentation-Preset (light/medium/heavy)
        patience: Early Stopping Patience (default: 50)
        model_name: Optional - Custom Name für das Modell
    """
    data = request.get_json()

    export_path = data.get("export_path")
    base_model = data.get("base_model", "yolov8n.pt")
    epochs = data.get("epochs", 100)
    batch_size = data.get("batch_size", 16)
    image_size = data.get("image_size", 1080)
    train_split = data.get("train_split", 0.8)
    val_split = data.get("val_split", 0.2)
    augmentation = data.get("augmentation", "medium")
    patience = data.get("patience", 50)
    custom_name = data.get("model_name", "").strip()

    # Validierung
    if not export_path:
        return jsonify({"error": "Export-Pfad erforderlich"}), 400

    export_dir = Path(export_path)
    if not export_dir.exists():
        return jsonify({"error": f"Export-Ordner nicht gefunden: {export_path}"}), 404

    data_yaml = export_dir / "data.yaml"
    if not data_yaml.exists():
        return jsonify({"error": "data.yaml nicht im Export-Ordner gefunden"}), 400

    # Augmentation-Preset laden
    config = load_yolo_config()
    aug_config = config.get("augmentation_presets", {}).get(augmentation, {})

    # Training-Run erstellen
    run_count = db.session.query(TrainingRun).count()
    model_name = custom_name or f"floorball_{run_count + 1}"

    hyperparameters = {
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "patience": patience,
        "train_split": train_split,
        "val_split": val_split,
        "augmentation": augmentation,
        **aug_config  # Augmentation-Parameter hinzufügen
    }

    run = TrainingRun(
        model_name=model_name,
        base_model=base_model,
        hyperparameters=hyperparameters,
        status="pending"
    )
    db.session.add(run)
    db.session.commit()

    # Celery Task starten
    task = start_training.delay(
        str(run.id),
        str(export_path),
        base_model,
        hyperparameters
    )

    run.celery_task_id = task.id
    db.session.commit()

    return jsonify({
        "success": True,
        "training_id": str(run.id),
        "task_id": task.id,
        "model_name": model_name
    })


@training_bp.route("/api/retrain/<run_id>", methods=["POST"])
def api_retrain(run_id):
    """
    Startet ein Training mit den gleichen Einstellungen wie ein vorheriges.

    Args:
        run_id: ID des vorherigen Training-Runs
    """
    old_run = db.session.get(TrainingRun, run_id)
    if not old_run:
        return jsonify({"error": "Training Run nicht gefunden"}), 404

    # Hyperparameter kopieren oder Defaults verwenden
    if old_run.hyperparameters:
        hyperparameters = old_run.hyperparameters.copy()
    else:
        # Fallback zu Defaults
        config = load_yolo_config()
        defaults = config.get("training_defaults", {})
        hyperparameters = {
            "epochs": defaults.get("epochs", 100),
            "batch": defaults.get("batch_size", 16),
            "imgsz": defaults.get("image_size", 1080),
            "patience": defaults.get("patience", 50),
            "train_split": 0.8,
        }

    # Finde den Export-Pfad
    exports = get_available_exports()
    if not exports:
        return jsonify({"error": "Keine exportierten Datasets gefunden. Bitte zuerst Labels exportieren."}), 400

    # Verwende den ersten verfügbaren Export
    export_path = exports[0]["path"]

    # Neuen Run-Namen generieren
    run_count = db.session.query(TrainingRun).count()
    # Entferne bestehende Versionsnummern wie _v2, _v3, etc.
    base_name = old_run.model_name
    base_name = re.sub(r'_v\d+$', '', base_name)
    model_name = f"{base_name}_v{run_count + 1}"

    # Neuen Training-Run erstellen
    new_run = TrainingRun(
        model_name=model_name,
        base_model=old_run.base_model,
        hyperparameters=hyperparameters,
        status="pending"
    )
    db.session.add(new_run)
    db.session.commit()

    # Celery Task starten
    try:
        task = start_training.delay(
            str(new_run.id),
            str(export_path),
            old_run.base_model,
            hyperparameters
        )

        new_run.celery_task_id = task.id
        db.session.commit()

        return jsonify({
            "success": True,
            "training_id": str(new_run.id),
            "task_id": task.id,
            "model_name": model_name
        })
    except Exception as e:
        new_run.status = "failed"
        new_run.error_message = str(e)
        db.session.commit()
        return jsonify({"error": f"Fehler beim Starten des Trainings: {str(e)}"}), 500


@training_bp.route("/api/models")
def api_list_models():
    """Gibt alle verfügbaren YOLO-Modelle zurück."""
    config = load_yolo_config()
    return jsonify({
        "models": config.get("models", {}),
        "recommendations": config.get("recommendations", {}),
        "training_defaults": config.get("training_defaults", {}),
        "augmentation_presets": config.get("augmentation_presets", {})
    })


@training_bp.route("/api/exports")
def api_list_exports():
    """Gibt alle verfügbaren Exports zurück."""
    return jsonify(get_available_exports())


@training_bp.route("/api/runs")
def api_list_runs():
    """Listet alle Training-Runs."""
    runs = db.session.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()

    return jsonify([{
        "id": str(r.id),
        "model_name": r.model_name,
        "base_model": r.base_model,
        "status": r.status,
        "final_map50": r.final_map50,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
    } for r in runs])


@training_bp.route("/api/runs/<run_id>")
def api_get_run(run_id):
    """Holt einen Training-Run mit Details."""
    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    return jsonify({
        "id": str(run.id),
        "model_name": run.model_name,
        "base_model": run.base_model,
        "hyperparameters": run.hyperparameters,
        "status": run.status,
        "error_message": run.error_message,
        "model_path": run.model_path,
        "final_map50": run.final_map50,
        "final_map50_95": run.final_map50_95,
        "training_time": run.training_time,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    })


@training_bp.route("/api/runs/<run_id>/status")
def api_run_status(run_id):
    """
    Holt den aktuellen Status eines Runs.
    Wird für Polling während des Trainings verwendet.
    """
    import csv

    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    # Celery Task Status prüfen
    task_info = None
    celery_task_id = getattr(run, 'celery_task_id', None)
    if celery_task_id:
        try:
            from celery.result import AsyncResult
            from src.web.extensions import celery
            result = AsyncResult(celery_task_id, app=celery)
            # Info nur als String speichern (kann Exception-Objekte enthalten)
            info_str = None
            if result.info:
                if isinstance(result.info, dict):
                    info_str = result.info
                elif isinstance(result.info, Exception):
                    info_str = {"error": str(result.info)}
                else:
                    info_str = str(result.info)
            task_info = {
                "state": result.state,
                "info": info_str
            }

            # Wenn Task FAILURE oder Worker lost, Run als abgebrochen markieren
            if result.state in ("FAILURE", "REVOKED") or (result.info and isinstance(result.info, Exception)):
                if run.status == "running":
                    run.status = "cancelled"
                    run.error_message = str(result.info) if result.info else "Task abgebrochen"
                    db.session.commit()
        except Exception:
            pass

    # Wenn Run "running" ist aber kein aktiver Celery Task existiert, prüfen ob wirklich noch läuft
    if run.status == "running" and celery_task_id:
        try:
            from celery.result import AsyncResult
            from src.web.extensions import celery
            result = AsyncResult(celery_task_id, app=celery)
            # PENDING ohne Info bedeutet Task existiert nicht mehr
            if result.state == "PENDING" and result.info is None:
                # Prüfe ob results.csv existiert und sich ändert
                results_csv = Path(f"models/trained/{run.model_name}/results.csv")
                if not results_csv.exists():
                    # Kein Progress, Task wahrscheinlich abgebrochen
                    run.status = "cancelled"
                    run.error_message = "Worker-Verbindung verloren"
                    db.session.commit()
        except Exception:
            pass

    # Progress aus results.csv lesen (YOLO schreibt diese während des Trainings)
    progress_info = None
    if run.status == "running":
        results_csv = Path(f"models/trained/{run.model_name}/results.csv")
        if results_csv.exists():
            try:
                with open(results_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        current_epoch = len(rows)
                        total_epochs = run.hyperparameters.get("epochs", 100) if run.hyperparameters else 100
                        progress_info = {
                            "epoch": current_epoch,
                            "total_epochs": total_epochs,
                            "progress": round((current_epoch / total_epochs) * 100, 1)
                        }
                        # Letzte Metriken
                        last_row = rows[-1]
                        for col in ["metrics/mAP50(B)", "mAP50", "mAP_0.5"]:
                            col_stripped = col.strip()
                            matching_cols = [k for k in last_row.keys() if col_stripped in k.strip()]
                            if matching_cols:
                                try:
                                    progress_info["current_map50"] = float(last_row[matching_cols[0]].strip())
                                    break
                                except:
                                    pass
            except Exception as e:
                print(f"Error reading results.csv: {e}")

    return jsonify({
        "status": run.status,
        "task": task_info,
        "progress": progress_info,
        "final_map50": run.final_map50,
    })


@training_bp.route("/api/activate/<run_id>", methods=["POST"])
def api_activate_model(run_id):
    """Aktiviert ein trainiertes Modell."""
    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    if run.status != "completed" or not run.model_path:
        return jsonify({"error": "Modell nicht bereit"}), 400

    # Altes aktives Modell deaktivieren
    old_active = db.session.query(ActiveModel).first()
    if old_active:
        db.session.delete(old_active)

    # Neues Modell aktivieren
    active = ActiveModel(
        training_run_id=run.id,
        model_path=run.model_path,
        model_name=run.model_name,
        map50=run.final_map50
    )
    db.session.add(active)
    db.session.commit()

    return jsonify({
        "success": True,
        "model_name": run.model_name,
        "model_path": run.model_path
    })


@training_bp.route("/api/active")
def api_get_active_model():
    """Gibt das aktive Modell zurück."""
    active = db.session.query(ActiveModel).first()

    if not active:
        return jsonify({"active": False})

    return jsonify({
        "active": True,
        "model_name": active.model_name,
        "model_path": active.model_path,
        "map50": active.map50,
        "activated_at": active.activated_at.isoformat() if active.activated_at else None
    })


@training_bp.route("/api/runs/<run_id>/logs")
def api_run_logs(run_id):
    """Gibt die letzten Zeilen des Celery-Logs zurück."""
    import subprocess

    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    # Celery Log lesen
    log_lines = []
    try:
        result = subprocess.run(
            ["tail", "-n", "20", "/tmp/celery.log"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            log_lines = result.stdout.strip().split('\n')
    except Exception as e:
        log_lines = [f"Fehler beim Lesen der Logs: {e}"]

    # GPU-Status prüfen
    gpu_info = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                gpu_info = {
                    "utilization": int(parts[0]),
                    "memory_used": int(parts[1]),
                    "memory_total": int(parts[2]),
                    "temperature": int(parts[3])
                }
    except Exception:
        pass

    return jsonify({
        "logs": log_lines,
        "gpu": gpu_info
    })


@training_bp.route("/api/runs/<run_id>/cancel", methods=["POST"])
def api_cancel_run(run_id):
    """Bricht ein laufendes Training ab."""
    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    if run.status != "running":
        return jsonify({"error": "Training läuft nicht"}), 400

    # Celery Task abbrechen
    if run.celery_task_id:
        try:
            from src.web.extensions import celery
            celery.control.revoke(run.celery_task_id, terminate=True, signal='SIGKILL')
        except Exception as e:
            print(f"Error revoking task: {e}")

    run.status = "cancelled"
    run.error_message = "Manuell abgebrochen"
    run.completed_at = datetime.utcnow()
    db.session.commit()

    return jsonify({"success": True, "message": f"Training '{run.model_name}' abgebrochen"})


@training_bp.route("/api/runs/<run_id>", methods=["DELETE"])
def api_delete_run(run_id):
    """Löscht einen Training-Run und optional die zugehörigen Dateien."""
    import shutil

    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    # Aktives Modell kann nicht gelöscht werden
    active = db.session.query(ActiveModel).filter_by(training_run_id=run.id).first()
    if active:
        return jsonify({"error": "Aktives Modell kann nicht gelöscht werden. Bitte zuerst deaktivieren."}), 400

    # Laufendes Training abbrechen
    if run.status == "running" and run.celery_task_id:
        try:
            from src.web.extensions import celery
            celery.control.revoke(run.celery_task_id, terminate=True)
        except Exception:
            pass

    # Modell-Ordner löschen falls vorhanden
    model_dir = Path(f"models/trained/{run.model_name}")
    if model_dir.exists():
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            print(f"Fehler beim Löschen von {model_dir}: {e}")

    # Aus DB löschen
    db.session.delete(run)
    db.session.commit()

    return jsonify({"success": True, "message": f"Training '{run.model_name}' gelöscht"})
