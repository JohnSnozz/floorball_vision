"""YOLO Training Script for Floorball Vision.

Train YOLO models for floorball object detection.
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def get_best_device() -> str:
    """Erkennt automatisch die beste verfügbare Hardware (GPU/CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU erkannt: {gpu_name}")
            return "0"  # Erste GPU verwenden
    except Exception as e:
        print(f"GPU-Erkennung fehlgeschlagen: {e}")

    print("Keine GPU verfügbar, verwende CPU")
    return "cpu"


def train_model(
    data_yaml: str,
    model: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 1080,
    batch: int = 4,
    project: str = 'runs/detect',
    name: str = 'train',
    device: Optional[str] = None,
    resume: bool = False,
    patience: int = 50,
    **kwargs
) -> str:
    """
    Train a YOLO model.

    Args:
        data_yaml: Path to data.yaml configuration
        model: Base model to use (e.g., 'yolov8n.pt', 'yolov5l6u.pt')
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project: Project directory for outputs
        name: Run name
        device: Device to use ('cuda', 'cpu', or specific GPU). Auto-detected if None.
        resume: Resume from last checkpoint
        patience: Early stopping patience (epochs without improvement)
        **kwargs: Additional training arguments (augmentation params, etc.)

    Returns:
        Path to best weights file
    """
    # GPU automatisch erkennen wenn nicht explizit gesetzt
    if device is None:
        device = get_best_device()

    yolo = YOLO(model)

    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'project': project,
        'name': name,
        'resume': resume,
        'patience': patience,
        'exist_ok': True,  # Überschreibe bestehende Runs
        'workers': 0,  # Wichtig für Celery - keine Sub-Prozesse
        'device': device,  # GPU/CPU
    }

    # Zusätzliche Parameter (Augmentation etc.) hinzufügen
    for key, value in kwargs.items():
        if value is not None:
            train_args[key] = value

    print(f"Starting training with device: {device}")
    results = yolo.train(**train_args)

    weights_path = Path(project) / name / 'weights' / 'best.pt'
    print(f"Training complete. Best weights: {weights_path}")

    return str(weights_path)


def validate_model(
    model_path: str,
    data_yaml: str,
    imgsz: int = 1080
) -> dict:
    """
    Validate a trained model.

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml configuration
        imgsz: Image size for validation

    Returns:
        Validation metrics
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz)

    return {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr
    }


def predict(
    model_path: str,
    source: str,
    output_dir: str = 'runs/detect/predict',
    conf: float = 0.25,
    save: bool = True
) -> list:
    """
    Run inference on images or video.

    Args:
        model_path: Path to trained model weights
        source: Path to image, video, or directory
        output_dir: Directory to save results
        conf: Confidence threshold
        save: Save results to disk

    Returns:
        List of detection results
    """
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project=output_dir
    )

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLO model')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', required=True, help='Path to data.yaml')
    train_parser.add_argument('--model', default='yolov8n.pt', help='Base model')
    train_parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    train_parser.add_argument('--imgsz', type=int, default=1080, help='Image size')
    train_parser.add_argument('--batch', type=int, default=4, help='Batch size')
    train_parser.add_argument('--project', default='runs/detect', help='Project dir')
    train_parser.add_argument('--name', default='train', help='Run name')
    train_parser.add_argument('--device', help='Device (cuda/cpu)')
    train_parser.add_argument('--resume', action='store_true', help='Resume training')

    val_parser = subparsers.add_parser('validate', help='Validate model')
    val_parser.add_argument('--model', required=True, help='Model weights path')
    val_parser.add_argument('--data', required=True, help='Path to data.yaml')
    val_parser.add_argument('--imgsz', type=int, default=1080, help='Image size')

    pred_parser = subparsers.add_parser('predict', help='Run inference')
    pred_parser.add_argument('--model', required=True, help='Model weights path')
    pred_parser.add_argument('--source', required=True, help='Input source')
    pred_parser.add_argument('--output', default='runs/detect/predict', help='Output dir')
    pred_parser.add_argument('--conf', type=float, default=0.25, help='Confidence')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(
            args.data, args.model, args.epochs, args.imgsz,
            args.batch, args.project, args.name, args.device, args.resume
        )
    elif args.command == 'validate':
        metrics = validate_model(args.model, args.data, args.imgsz)
        print(f"Validation metrics: {metrics}")
    elif args.command == 'predict':
        predict(args.model, args.source, args.output, args.conf)


if __name__ == '__main__':
    main()
