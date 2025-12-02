#!/usr/bin/env python3
"""
Generiert YOLO Predictions für alle Tasks in einem Label Studio Projekt
und importiert sie direkt als Predictions.

Usage:
    python scripts/create_predictions.py --project 1 --model models/trained/fb_large_20/weights/best.pt
"""
import os
import sys
import argparse
import requests
from pathlib import Path

# Projekt-Root zu Python-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def get_label_studio_tasks(project_id: int, api_key: str, base_url: str) -> list:
    """Holt alle Tasks aus einem Label Studio Projekt."""
    headers = {"Authorization": f"Token {api_key}"}

    all_tasks = []
    page = 1
    page_size = 100

    while True:
        resp = requests.get(
            f"{base_url}/api/projects/{project_id}/tasks",
            params={"page": page, "page_size": page_size},
            headers=headers
        )

        if not resp.ok:
            print(f"Error fetching tasks: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        tasks = data.get("tasks", data) if isinstance(data, dict) else data

        if not tasks:
            break

        all_tasks.extend(tasks)

        # Prüfen ob es weitere Seiten gibt
        if isinstance(data, dict) and data.get("next"):
            page += 1
        elif len(tasks) < page_size:
            break
        else:
            page += 1

    return all_tasks


def resolve_image_path(url: str, base_url: str = None) -> str:
    """Löst Label Studio URL zu lokalem Pfad auf."""
    from urllib.parse import unquote

    if not url:
        return None

    url = unquote(url)

    # Absoluter Pfad
    if os.path.isabs(url) and os.path.exists(url):
        return url

    # Label Studio upload Format
    if '/data/upload/' in url:
        parts = url.split('/data/upload/')[-1]
        possible_paths = [
            f"/home/jonas/.local/share/label-studio/media/upload/{parts}",
            os.path.expanduser(f"~/.local/share/label-studio/media/upload/{parts}"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p

    return None


def run_yolo_prediction(image_path: str, model, confidence: float = 0.25) -> list:
    """Führt YOLO Prediction durch und konvertiert zu Label Studio Format."""
    results = model.predict(image_path, conf=confidence, verbose=False)

    if not results or not results[0].boxes:
        return []

    prediction = results[0]
    annotations = []

    img_height, img_width = prediction.orig_shape

    for i, box in enumerate(prediction.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x_percent = (x1 / img_width) * 100
        y_percent = (y1 / img_height) * 100
        width_percent = ((x2 - x1) / img_width) * 100
        height_percent = ((y2 - y1) / img_height) * 100

        class_id = int(box.cls[0])
        class_name = model.names.get(class_id, f"class_{class_id}")
        confidence_score = float(box.conf[0])

        annotations.append({
            "id": f"yolo_{i}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": img_width,
            "original_height": img_height,
            "value": {
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rotation": 0,
                "rectanglelabels": [class_name]
            },
            "score": confidence_score
        })

    return annotations


def create_prediction(task_id: int, result: list, model_version: str,
                      api_key: str, base_url: str) -> bool:
    """Erstellt eine Prediction in Label Studio."""
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "task": task_id,
        "result": result,
        "model_version": model_version,
        "score": sum(r.get("score", 0) for r in result) / len(result) if result else 0
    }

    resp = requests.post(
        f"{base_url}/api/predictions",
        json=payload,
        headers=headers
    )

    return resp.ok


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO predictions for Label Studio")
    parser.add_argument("--project", "-p", type=int, required=True, help="Label Studio project ID")
    parser.add_argument("--model", "-m", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")

    args = parser.parse_args()

    # Environment laden
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    base_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")

    if not api_key:
        print("Error: LABEL_STUDIO_API_KEY not set in .env")
        sys.exit(1)

    # YOLO Modell laden
    print(f"Loading YOLO model from {args.model}...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    model_version = os.path.basename(args.model)
    print(f"Model loaded. Classes: {model.names}")

    # Tasks holen
    print(f"\nFetching tasks from project {args.project}...")
    tasks = get_label_studio_tasks(args.project, api_key, base_url)
    print(f"Found {len(tasks)} tasks")

    if not tasks:
        print("No tasks found!")
        sys.exit(0)

    # Predictions generieren
    success_count = 0
    error_count = 0
    skip_count = 0

    for i, task in enumerate(tasks):
        task_id = task.get("id")
        image_url = task.get("data", {}).get("image", "")

        # Prüfen ob schon Predictions existieren
        existing_predictions = task.get("predictions", [])
        if existing_predictions and not args.overwrite:
            skip_count += 1
            continue

        # Bild-Pfad auflösen
        image_path = resolve_image_path(image_url, base_url)

        if not image_path or not os.path.exists(image_path):
            print(f"  [{i+1}/{len(tasks)}] Task {task_id}: Image not found - {image_url}")
            error_count += 1
            continue

        # Prediction durchführen
        try:
            annotations = run_yolo_prediction(image_path, model, args.confidence)

            # In Label Studio speichern
            if create_prediction(task_id, annotations, model_version, api_key, base_url):
                print(f"  [{i+1}/{len(tasks)}] Task {task_id}: {len(annotations)} detections")
                success_count += 1
            else:
                print(f"  [{i+1}/{len(tasks)}] Task {task_id}: Failed to save prediction")
                error_count += 1

        except Exception as e:
            print(f"  [{i+1}/{len(tasks)}] Task {task_id}: Error - {e}")
            error_count += 1

    print(f"\n=== Summary ===")
    print(f"Success: {success_count}")
    print(f"Errors:  {error_count}")
    print(f"Skipped: {skip_count}")


if __name__ == "__main__":
    main()
