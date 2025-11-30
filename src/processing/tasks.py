"""
Celery Tasks

Background Tasks für Video-Verarbeitung und Training.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

from src.web.extensions import db, celery
from src.db.models import Video, TrainingRun


def get_flask_app():
    """Flask App für Celery Context erstellen."""
    from src.web.app import create_app
    return create_app()


@celery.task(bind=True)
def download_youtube_video(self, video_id: str, url: str):
    """
    YouTube Video herunterladen.

    Args:
        video_id: UUID des Videos in der DB
        url: YouTube URL
    """
    from src.processing.downloader import download_video, DownloadError
    from src.web.config import BaseConfig

    app = get_flask_app()
    with app.app_context():
        video = db.session.get(Video, video_id)
        if not video:
            return {"error": "Video nicht gefunden"}

        try:
            # Download-Pfad
            videos_path = BaseConfig.VIDEOS_PATH
            os.makedirs(videos_path, exist_ok=True)

            # Video herunterladen
            result = download_video(
                url=url,
                output_dir=videos_path,
                video_id=video_id,
                progress_callback=lambda p: self.update_state(
                    state="PROGRESS",
                    meta={"progress": p}
                )
            )

            # Video in DB aktualisieren
            video.filename = result["filename"]
            video.original_filename = result["title"]
            video.file_path = result["file_path"]
            video.duration = result.get("duration")
            video.status = "processing"
            db.session.commit()

            # Metadaten extrahieren
            process_uploaded_video.delay(video_id)

            return {"status": "success", "video_id": video_id}

        except DownloadError as e:
            video.status = "error"
            video.error_message = str(e)
            db.session.commit()
            return {"error": str(e)}

        except Exception as e:
            video.status = "error"
            video.error_message = f"Unbekannter Fehler: {str(e)}"
            db.session.commit()
            raise


@celery.task
def process_uploaded_video(video_id: str):
    """
    Hochgeladenes Video verarbeiten.

    - Metadaten extrahieren
    - Thumbnail generieren

    Args:
        video_id: UUID des Videos
    """
    from src.utils.video_utils import get_video_metadata, generate_thumbnail

    app = get_flask_app()
    with app.app_context():
        video = db.session.get(Video, video_id)
        if not video:
            return {"error": "Video nicht gefunden"}

        try:
            # Metadaten extrahieren
            if video.file_path and os.path.exists(video.file_path):
                metadata = get_video_metadata(video.file_path)
                video.duration = metadata.get("duration")
                video.fps = metadata.get("fps")
                video.width = metadata.get("width")
                video.height = metadata.get("height")
                video.codec = metadata.get("codec")

                # Thumbnail generieren
                thumb_path = video.file_path.rsplit(".", 1)[0] + "_thumb.jpg"
                generate_thumbnail(video.file_path, thumb_path)

            video.status = "ready"
            db.session.commit()

            return {"status": "success", "video_id": video_id}

        except Exception as e:
            video.status = "error"
            video.error_message = f"Verarbeitung fehlgeschlagen: {str(e)}"
            db.session.commit()
            return {"error": str(e)}


def prepare_train_val_split(export_path: str, train_split: float = 0.8) -> str:
    """
    Erstellt Train/Val Split für YOLO Training.

    Kopiert Bilder und Labels in train/ und val/ Unterordner
    und erstellt ein neues data.yaml.

    Args:
        export_path: Pfad zum Export-Ordner
        train_split: Anteil Training (0.0-1.0)

    Returns:
        Pfad zum neuen data.yaml
    """
    import random
    import yaml

    export_dir = Path(export_path)
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"

    # Alle Bilder finden
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not images:
        raise ValueError(f"Keine Bilder in {images_dir}")

    # Zufällig mischen und aufteilen
    random.shuffle(images)
    split_idx = int(len(images) * train_split)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Train/Val Ordner erstellen
    train_img_dir = export_dir / "train" / "images"
    train_lbl_dir = export_dir / "train" / "labels"
    val_img_dir = export_dir / "val" / "images"
    val_lbl_dir = export_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Bilder und Labels kopieren
    def copy_to_split(image_list, img_dest, lbl_dest):
        for img_path in image_list:
            # Bild kopieren
            shutil.copy2(img_path, img_dest / img_path.name)

            # Label kopieren falls vorhanden
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name
            if label_path.exists():
                shutil.copy2(label_path, lbl_dest / label_name)

    copy_to_split(train_images, train_img_dir, train_lbl_dir)
    copy_to_split(val_images, val_img_dir, val_lbl_dir)

    # Klassen lesen
    classes_file = export_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = ["ball", "cornerpoints", "goal", "goalkeeper",
                   "period", "player", "ref", "scoreboard", "time"]

    # Neues data.yaml erstellen
    data_yaml_content = {
        "path": str(export_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(classes)}
    }

    data_yaml_path = export_dir / "data_split.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)

    return str(data_yaml_path)


@celery.task(bind=True)
def start_training(self, run_id: str, export_path: str, base_model: str, hyperparameters: dict):
    """
    YOLO Training starten.

    Args:
        run_id: TrainingRun UUID
        export_path: Pfad zum exportierten Dataset
        base_model: Basis-Modell (z.B. yolov8n.pt)
        hyperparameters: Training Parameter (epochs, batch, imgsz, etc.)
    """
    from src.training.train import train_model

    app = get_flask_app()
    with app.app_context():
        run = db.session.get(TrainingRun, run_id)
        if not run:
            return {"error": "Training Run nicht gefunden"}

        try:
            run.status = "running"
            run.started_at = datetime.utcnow()
            db.session.commit()

            # Parameter extrahieren
            epochs = hyperparameters.get("epochs", 100)
            batch_size = hyperparameters.get("batch", 16)
            image_size = hyperparameters.get("imgsz", 1080)
            patience = hyperparameters.get("patience", 50)
            train_split = hyperparameters.get("train_split", 0.8)

            # Train/Val Split erstellen
            self.update_state(state="PROGRESS", meta={"stage": "preparing", "message": "Erstelle Train/Val Split..."})

            data_yaml = prepare_train_val_split(export_path, train_split)

            # Training starten
            self.update_state(state="PROGRESS", meta={"stage": "training", "epoch": 0, "total_epochs": epochs})

            # Augmentation Parameter extrahieren (falls vorhanden)
            aug_params = {}
            aug_keys = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                        "fliplr", "mosaic", "mixup", "copy_paste"]
            for key in aug_keys:
                if key in hyperparameters:
                    aug_params[key] = hyperparameters[key]

            model_path = train_model(
                data_yaml=data_yaml,
                model=base_model,
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                project="models/trained",
                name=run.model_name,
                **aug_params
            )

            # Modell-Pfad
            final_model_path = f"models/trained/{run.model_name}/weights/best.pt"

            # Falls best.pt nicht existiert, last.pt verwenden
            if not Path(final_model_path).exists():
                alt_path = f"models/trained/{run.model_name}/weights/last.pt"
                if Path(alt_path).exists():
                    final_model_path = alt_path

            # Metriken aus results.csv lesen falls vorhanden
            results_csv = Path(f"models/trained/{run.model_name}/results.csv")
            if results_csv.exists():
                import csv
                with open(results_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        # Spalten können variieren, versuche verschiedene Namen
                        for col in ["metrics/mAP50(B)", "mAP50", "mAP_0.5"]:
                            if col in last_row:
                                try:
                                    run.final_map50 = float(last_row[col].strip())
                                    break
                                except:
                                    pass
                        for col in ["metrics/mAP50-95(B)", "mAP50-95", "mAP_0.5:0.95"]:
                            if col in last_row:
                                try:
                                    run.final_map50_95 = float(last_row[col].strip())
                                    break
                                except:
                                    pass

            run.model_path = final_model_path
            run.status = "completed"
            run.completed_at = datetime.utcnow()

            # Trainingszeit berechnen
            if run.started_at:
                run.training_time = (run.completed_at - run.started_at).total_seconds()

            db.session.commit()

            return {
                "status": "success",
                "run_id": run_id,
                "model_path": final_model_path,
                "map50": run.final_map50
            }

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.session.commit()
            return {"error": str(e)}
