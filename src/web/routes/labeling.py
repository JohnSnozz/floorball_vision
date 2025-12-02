"""
Labeling Routes

Endpoints für Label Studio Integration.
Option A: Ein Projekt kann mehrere Batches aus verschiedenen Videos haben.
"""
from flask import Blueprint, render_template, request, jsonify, abort
from src.web.extensions import db
from src.db.models import Video, LabelingProject, LabelingBatch
from src.labeling import LabelStudioClient, LabelingSync, YOLOExporter, LabelStudioError

labeling_bp = Blueprint("labeling", __name__, url_prefix="/labeling")


# === Seiten ===

@labeling_bp.route("/")
def index():
    """Labeling Übersicht."""
    videos = db.session.query(Video).filter_by(status="ready").all()
    projects = db.session.query(LabelingProject).order_by(LabelingProject.created_at.desc()).all()

    # Label Studio Status prüfen
    try:
        client = LabelStudioClient()
        ls_status = client.health_check()
    except:
        ls_status = False

    return render_template(
        "labeling/index.html",
        videos=videos,
        projects=projects,
        label_studio_status=ls_status
    )


@labeling_bp.route("/datasets")
def datasets():
    """Dataset-Verwaltung (exportierte Labels)."""
    return render_template("labeling/datasets.html")


@labeling_bp.route("/project/<project_id>")
def project_detail(project_id):
    """Projekt-Details mit Batches."""
    project = db.session.get(LabelingProject, project_id)
    if not project:
        abort(404)

    videos = db.session.query(Video).filter_by(status="ready").all()

    # Label Studio Infos holen
    ls_info = None
    try:
        client = LabelStudioClient()
        tasks = client.list_tasks(project.label_studio_id)

        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.get("annotations")])

        ls_info = {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        }
    except:
        pass

    return render_template(
        "labeling/project.html",
        project=project,
        videos=videos,
        ls_info=ls_info
    )


# === API Endpoints ===

@labeling_bp.route("/api/status")
def api_status():
    """Label Studio Status."""
    try:
        client = LabelStudioClient()
        healthy = client.health_check()
        return jsonify({
            "status": "online" if healthy else "offline",
            "url": client.url
        })
    except LabelStudioError as e:
        return jsonify({"status": "error", "message": str(e)})


@labeling_bp.route("/api/create-project", methods=["POST"])
def api_create_project():
    """
    Erstellt ein neues leeres Projekt.

    JSON Body:
        title: Projekt-Titel
    """
    data = request.get_json()
    title = data.get("title", "").strip()

    if not title:
        return jsonify({"error": "Titel erforderlich"}), 400

    try:
        # Label Studio Projekt erstellen
        client = LabelStudioClient()
        ls_project = client.create_project(
            title=title,
            description="Floorball Vision Training Data"
        )

        # In DB speichern
        project = LabelingProject(
            label_studio_id=ls_project["id"],
            title=title,
            status="created"
        )
        db.session.add(project)
        db.session.commit()

        return jsonify({
            "success": True,
            "project_id": str(project.id),
            "label_studio_id": ls_project["id"],
            "label_studio_url": client.get_labeling_url(ls_project["id"]),
        })

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/link-project", methods=["POST"])
def api_link_project():
    """
    Verknüpft ein bestehendes Label Studio Projekt.

    JSON Body:
        label_studio_id: ID des Label Studio Projekts
        title: Optional - Titel (sonst aus Label Studio übernommen)
    """
    data = request.get_json()
    ls_id = data.get("label_studio_id")

    if not ls_id:
        return jsonify({"error": "label_studio_id erforderlich"}), 400

    # Prüfen ob bereits verknüpft
    existing = db.session.query(LabelingProject).filter_by(label_studio_id=ls_id).first()
    if existing:
        return jsonify({
            "error": f"Projekt bereits verknüpft als '{existing.title}'",
            "project_id": str(existing.id)
        }), 400

    try:
        client = LabelStudioClient()
        ls_project = client.get_project(ls_id)

        title = data.get("title", "").strip() or ls_project.get("title", f"Projekt {ls_id}")

        # In DB speichern
        project = LabelingProject(
            label_studio_id=ls_id,
            title=title,
            status="labeling"  # Da es bereits existiert
        )
        db.session.add(project)
        db.session.commit()

        return jsonify({
            "success": True,
            "project_id": str(project.id),
            "label_studio_id": ls_id,
            "title": title,
            "label_studio_url": client.get_labeling_url(ls_id),
        })

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/label-studio-projects")
def api_list_label_studio_projects():
    """Listet alle Projekte in Label Studio (zum Verknüpfen)."""
    try:
        client = LabelStudioClient()
        ls_response = client.list_projects()

        # Label Studio gibt ein Dict mit 'results' zurück
        ls_projects = ls_response.get("results", []) if isinstance(ls_response, dict) else ls_response

        # Bereits verknüpfte IDs
        linked_ids = {p.label_studio_id for p in db.session.query(LabelingProject).all()}

        result = []
        for p in ls_projects:
            pid = p.get("id")
            result.append({
                "id": pid,
                "title": p.get("title"),
                "task_count": p.get("task_number", 0),
                "linked": pid in linked_ids
            })

        return jsonify(result)

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/add-batch", methods=["POST"])
def api_add_batch():
    """
    Fügt einen Batch von Frames zu einem bestehenden Projekt hinzu.

    JSON Body:
        project_id: Projekt UUID
        video_id: Video UUID
        num_frames: Anzahl Frames (default: 100)
    """
    data = request.get_json()

    project_id = data.get("project_id")
    video_id = data.get("video_id")
    num_frames = data.get("num_frames", 100)

    if not project_id:
        return jsonify({"error": "project_id erforderlich"}), 400
    if not video_id:
        return jsonify({"error": "video_id erforderlich"}), 400

    # Projekt laden
    project = db.session.get(LabelingProject, project_id)
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    # Video laden
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    if not video.file_path:
        return jsonify({"error": "Video hat keinen Dateipfad"}), 400

    try:
        sync = LabelingSync()

        # Frames extrahieren
        batch_result = sync.extract_random_frames(
            video_path=video.file_path,
            num_frames=num_frames,
            batch_id=f"v{str(video.id)[:8]}_{len(project.batches) + 1}"
        )

        # Frames zu Label Studio hochladen
        upload_result = sync.upload_batch_to_project(
            batch_id=batch_result["batch_id"],
            project_id=project.label_studio_id
        )

        # Batch in DB speichern
        batch = LabelingBatch(
            project_id=project.id,
            video_id=video.id,
            batch_id=batch_result["batch_id"],
            num_frames=batch_result["total_extracted"],
            frames_path=batch_result["batch_dir"],
            task_ids=upload_result.get("task_ids", []),
            status="uploaded"
        )
        db.session.add(batch)

        # Projekt-Status aktualisieren
        project.status = "labeling"
        db.session.commit()

        return jsonify({
            "success": True,
            "batch_id": batch_result["batch_id"],
            "frames_extracted": batch_result["total_extracted"],
            "frames_uploaded": upload_result["uploaded"],
            "video_name": video.original_filename or video.filename,
        })

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unbekannter Fehler: {str(e)}"}), 500


@labeling_bp.route("/api/add-batch-and-open", methods=["POST"])
def api_add_batch_and_open():
    """
    Erstellt einen Batch und gibt die Label Studio URL zurück.

    JSON Body:
        project_id: Projekt UUID
        video_id: Video UUID
        batch_name: Name für den Batch (optional)
        num_frames: Anzahl Frames (default: 100)

    Returns:
        URL zu Label Studio mit gefilterten Tasks
    """
    data = request.get_json()

    project_id = data.get("project_id")
    video_id = data.get("video_id")
    batch_name = data.get("batch_name", "").strip()
    num_frames = data.get("num_frames", 100)

    if not project_id:
        return jsonify({"error": "project_id erforderlich"}), 400
    if not video_id:
        return jsonify({"error": "video_id erforderlich"}), 400

    # Projekt laden
    project = db.session.get(LabelingProject, project_id)
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    # Video laden
    video = db.session.get(Video, video_id)
    if not video:
        return jsonify({"error": "Video nicht gefunden"}), 404

    if not video.file_path:
        return jsonify({"error": "Video hat keinen Dateipfad"}), 400

    # Batch-Name generieren falls nicht angegeben
    if not batch_name:
        # Kurzer Video-Name + Batch-Nummer
        video_short = (video.original_filename or video.filename or "video")[:20]
        video_short = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_short)
        batch_name = f"{video_short}_batch{len(project.batches) + 1}"

    # Name bereinigen
    batch_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in batch_name)

    try:
        sync = LabelingSync()

        # Frames extrahieren
        batch_result = sync.extract_random_frames(
            video_path=video.file_path,
            num_frames=num_frames,
            batch_id=batch_name
        )

        # Frames zu Label Studio hochladen
        upload_result = sync.upload_batch_to_project(
            batch_id=batch_result["batch_id"],
            project_id=project.label_studio_id
        )

        task_ids = upload_result.get("task_ids", [])

        # Label Studio View/Tab mit Filter erstellen für die neuen Tasks
        client = LabelStudioClient()
        view_id = None

        if task_ids:
            # Gefilterten View erstellen
            view = client.create_filtered_view(
                project_id=project.label_studio_id,
                title=batch_name,
                task_ids=task_ids
            )
            view_id = view.get("id")
            label_url = client.get_view_labeling_url(project.label_studio_id, view_id)
        else:
            label_url = client.get_labeling_url(project.label_studio_id)

        # Batch in DB speichern (mit View-ID)
        batch = LabelingBatch(
            project_id=project.id,
            video_id=video.id,
            batch_id=batch_name,
            num_frames=batch_result["total_extracted"],
            frames_path=batch_result["batch_dir"],
            task_ids=task_ids,
            view_id=view_id,
            status="uploaded"
        )
        db.session.add(batch)

        # Projekt-Status aktualisieren
        project.status = "labeling"
        db.session.commit()

        return jsonify({
            "success": True,
            "batch_id": batch_name,
            "frames_extracted": batch_result["total_extracted"],
            "frames_uploaded": upload_result["uploaded"],
            "video_name": video.original_filename or video.filename,
            "label_studio_url": label_url,
        })

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unbekannter Fehler: {str(e)}"}), 500


@labeling_bp.route("/api/batch/<batch_id>/open")
def api_open_batch(batch_id):
    """
    Gibt die Label Studio URL für einen spezifischen Batch zurück.
    Erstellt einen View/Tab falls noch keiner existiert.
    """
    batch = db.session.query(LabelingBatch).filter_by(batch_id=batch_id).first()
    if not batch:
        return jsonify({"error": "Batch nicht gefunden"}), 404

    project = batch.project
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    try:
        client = LabelStudioClient()

        # Prüfen ob View noch existiert (falls view_id vorhanden)
        view_exists = False
        if batch.view_id:
            try:
                # Versuche den View abzurufen
                client._request("GET", f"/dm/views/{batch.view_id}/")
                view_exists = True
            except:
                # View existiert nicht mehr
                batch.view_id = None

        if view_exists and batch.view_id:
            # Bestehenden View verwenden
            label_url = client.get_view_url(project.label_studio_id, batch.view_id)
        elif batch.task_ids and len(batch.task_ids) > 0:
            # View erstellen mit gespeicherten Task-IDs
            view = client.create_filtered_view(
                project_id=project.label_studio_id,
                title=batch.batch_id,
                task_ids=batch.task_ids
            )
            batch.view_id = view.get("id")
            db.session.commit()
            label_url = client.get_view_url(project.label_studio_id, batch.view_id)
        else:
            # Keine Task-IDs gespeichert - öffne Projekt ohne Tab-Filter
            label_url = client.get_labeling_url(project.label_studio_id)

            # Hinweis: Für ältere Batches ohne task_ids können wir keinen
            # gefilterten View erstellen

        return jsonify({
            "url": label_url,
            "batch_id": batch.batch_id,
            "num_frames": batch.num_frames,
            "has_view": batch.view_id is not None
        })

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/batch/<batch_id>/rename", methods=["POST"])
def api_rename_batch(batch_id):
    """
    Benennt einen Batch um.

    Aktualisiert auch den View-Titel in Label Studio falls vorhanden.

    JSON Body:
        new_name: Neuer Name für den Batch
    """
    batch = db.session.query(LabelingBatch).filter_by(batch_id=batch_id).first()
    if not batch:
        return jsonify({"error": "Batch nicht gefunden"}), 404

    data = request.get_json() or {}
    new_name = data.get("new_name", "").strip()

    if not new_name:
        return jsonify({"error": "Neuer Name erforderlich"}), 400

    # Name bereinigen
    new_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in new_name)

    if not new_name:
        return jsonify({"error": "Ungültiger Name"}), 400

    old_name = batch.batch_id

    try:
        # Label Studio View umbenennen falls vorhanden
        if batch.view_id and batch.project:
            try:
                client = LabelStudioClient()
                # View-Titel aktualisieren via PATCH
                client._request(
                    "PATCH",
                    f"/dm/views/{batch.view_id}/",
                    json={"data": {"title": new_name}}
                )
            except Exception as e:
                # View-Umbenennung fehlgeschlagen, aber lokale Umbenennung fortsetzen
                print(f"Warning: Could not rename Label Studio view: {e}")

        # Lokale Umbenennung
        batch.batch_id = new_name
        db.session.commit()

        return jsonify({
            "success": True,
            "old_name": old_name,
            "new_name": new_name
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/export/<project_id>", methods=["POST"])
def api_export_labels(project_id):
    """
    Exportiert Labels aus Label Studio.

    Holt die Annotations und speichert sie im YOLO-Format.

    JSON Body (optional):
        export_name: Name für das exportierte Dataset
    """
    project = db.session.get(LabelingProject, project_id)
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    # Export-Name aus Request oder Default
    data = request.get_json() or {}
    export_name = data.get("export_name", "").strip()

    # Wenn kein Name angegeben, Projekt-Titel verwenden
    if not export_name:
        export_name = project.title

    # Name bereinigen (nur alphanumerisch, Bindestrich, Unterstrich)
    export_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in export_name)

    # Sicherstellen dass der Name nicht leer ist
    if not export_name:
        export_name = f"dataset_{project.label_studio_id}"

    try:
        exporter = YOLOExporter()
        result = exporter.export_project(
            project_id=project.label_studio_id,
            export_name=export_name
        )

        # Status aktualisieren
        project.status = "exported"
        project.export_path = result["path"]
        db.session.commit()

        return jsonify({
            "success": True,
            "path": result["path"],
            "export_name": export_name,
            "images": result["images"],
            "labels": result["labels"],
            "classes": result["classes"],
        })

    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/projects")
def api_list_projects():
    """Listet alle Labeling-Projekte."""
    projects = db.session.query(LabelingProject).order_by(LabelingProject.created_at.desc()).all()

    result = []
    for p in projects:
        # Anzahl Frames aus allen Batches
        total_frames = sum(b.num_frames or 0 for b in p.batches)
        result.append({
            "id": str(p.id),
            "label_studio_id": p.label_studio_id,
            "title": p.title,
            "status": p.status,
            "batch_count": len(p.batches),
            "total_frames": total_frames,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        })

    return jsonify(result)


@labeling_bp.route("/api/projects/<project_id>")
def api_get_project(project_id):
    """Holt ein Projekt mit Details."""
    project = db.session.get(LabelingProject, project_id)
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    # Label Studio Infos
    ls_info = None
    try:
        client = LabelStudioClient()
        tasks = client.list_tasks(project.label_studio_id)

        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.get("annotations")])

        ls_info = {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        }
    except:
        pass

    # Batches
    batches = [{
        "id": str(b.id),
        "batch_id": b.batch_id,
        "num_frames": b.num_frames,
        "status": b.status,
        "created_at": b.created_at.isoformat() if b.created_at else None,
    } for b in project.batches]

    return jsonify({
        "id": str(project.id),
        "label_studio_id": project.label_studio_id,
        "title": project.title,
        "status": project.status,
        "export_path": project.export_path,
        "label_studio_info": ls_info,
        "batches": batches,
        "created_at": project.created_at.isoformat() if project.created_at else None,
    })


@labeling_bp.route("/api/open-label-studio/<project_id>")
def api_open_label_studio(project_id):
    """Gibt die Label Studio URL zurück."""
    project = db.session.get(LabelingProject, project_id)
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    try:
        client = LabelStudioClient()
        url = client.get_labeling_url(project.label_studio_id)
        return jsonify({"url": url})
    except LabelStudioError as e:
        return jsonify({"error": str(e)}), 500


@labeling_bp.route("/api/delete/<project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    """Löscht ein Labeling-Projekt."""
    project = db.session.get(LabelingProject, project_id)
    if not project:
        return jsonify({"error": "Projekt nicht gefunden"}), 404

    try:
        # Label Studio Projekt löschen
        client = LabelStudioClient()
        client.delete_project(project.label_studio_id)
    except:
        pass  # Ignorieren falls Label Studio nicht erreichbar

    # Batch-Dateien löschen
    for batch in project.batches:
        sync = LabelingSync()
        sync.delete_batch(batch.batch_id)

    # Aus DB löschen
    db.session.delete(project)
    db.session.commit()

    return jsonify({"success": True})
