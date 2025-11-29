"""
Training Routes

Endpoints für YOLO Training.
"""
import os
from flask import Blueprint, render_template, request, jsonify, abort
from src.web.extensions import db
from src.db.models import LabelingProject, TrainingRun, ActiveModel
from src.processing.tasks import start_training

training_bp = Blueprint("training", __name__, url_prefix="/training")


# === Seiten ===

@training_bp.route("/")
def index():
    """Training Übersicht."""
    runs = db.session.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
    active = db.session.query(ActiveModel).first()

    return render_template(
        "training/index.html",
        runs=runs,
        active_model=active
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
        dataset_ids: Liste von LabelingProject IDs (exportierte Datasets)
        base_model: Basis-Modell (default: yolov8n.pt)
        epochs: Anzahl Epochs (default: 50)
    """
    data = request.get_json()

    dataset_ids = data.get("dataset_ids", [])
    base_model = data.get("base_model", "yolov8n.pt")
    epochs = data.get("epochs", 50)

    if not dataset_ids:
        return jsonify({"error": "Mindestens ein Dataset erforderlich"}), 400

    # Datasets prüfen
    datasets = []
    for did in dataset_ids:
        project = db.session.get(LabelingProject, did)
        if not project:
            return jsonify({"error": f"Projekt {did} nicht gefunden"}), 404
        if project.status != "exported" or not project.export_path:
            return jsonify({"error": f"Projekt {project.title} nicht exportiert"}), 400
        datasets.append(project)

    # Training-Run erstellen
    run_count = db.session.query(TrainingRun).count()
    run = TrainingRun(
        labeling_project_id=datasets[0].id if len(datasets) == 1 else None,
        model_name=f"floorball_{run_count + 1}",
        base_model=base_model,
        hyperparameters={
            "epochs": epochs,
            "imgsz": 1080,
            "batch": 4,
        },
        status="pending"
    )
    db.session.add(run)
    db.session.commit()

    # Celery Task starten
    task = start_training.delay(
        str(run.id),
        [p.export_path for p in datasets],
        base_model,
        epochs
    )

    run.celery_task_id = task.id
    db.session.commit()

    return jsonify({
        "success": True,
        "training_id": str(run.id),
        "task_id": task.id
    })


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
    run = db.session.get(TrainingRun, run_id)
    if not run:
        return jsonify({"error": "Run nicht gefunden"}), 404

    # Celery Task Status prüfen
    task_info = None
    if run.celery_task_id:
        from celery.result import AsyncResult
        from src.web.extensions import celery
        result = AsyncResult(run.celery_task_id, app=celery)
        task_info = {
            "state": result.state,
            "info": result.info if result.info else None
        }

    return jsonify({
        "status": run.status,
        "task": task_info,
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
