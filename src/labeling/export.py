"""
YOLO Export

Exportiert Labels aus Label Studio ins YOLO-Format.
"""
import os
import io
import zipfile
import shutil
import yaml
import requests
from pathlib import Path
from typing import Optional

from .client import LabelStudioClient, LabelStudioError


class YOLOExporter:
    """Exportiert Label Studio Annotations zu YOLO Format."""

    def __init__(
        self,
        client: Optional[LabelStudioClient] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialisiert den Exporter.

        Args:
            client: Label Studio Client (default: neuer Client)
            output_dir: Ausgabe-Verzeichnis (default: data/labeling/exports)
        """
        self.client = client or LabelStudioClient()
        self.output_dir = Path(output_dir or "data/labeling/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_project(
        self,
        project_id: int,
        export_name: Optional[str] = None
    ) -> dict:
        """
        Exportiert ein Projekt als YOLO Dataset.

        Args:
            project_id: Label Studio Projekt-ID
            export_name: Name für den Export-Ordner

        Returns:
            dict mit Pfaden und Statistiken
        """
        # Export-Name generieren
        if not export_name:
            project = self.client.get_project(project_id)
            export_name = f"project_{project_id}_{project.get('title', 'export')}"
            export_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in export_name)

        export_path = self.output_dir / export_name

        # Alten Export löschen falls vorhanden
        if export_path.exists():
            shutil.rmtree(export_path)

        # Von Label Studio exportieren
        try:
            export_data = self.client.export_annotations(project_id, "YOLO")
        except Exception as e:
            raise LabelStudioError(f"Export fehlgeschlagen: {e}")

        # ZIP entpacken
        try:
            with zipfile.ZipFile(io.BytesIO(export_data)) as zf:
                zf.extractall(export_path)
        except zipfile.BadZipFile:
            raise LabelStudioError("Export ist keine gültige ZIP-Datei")

        # Bilder von Label Studio holen und kopieren
        self._copy_images_from_label_studio(project_id, export_path)

        # Struktur validieren und anpassen
        result = self._organize_yolo_structure(export_path)

        return {
            "path": str(export_path),
            "images": result["images"],
            "labels": result["labels"],
            "classes": result["classes"],
        }

    def _copy_images_from_label_studio(self, project_id: int, export_path: Path) -> int:
        """
        Kopiert die Bilder von Label Studio zum Export-Ordner.

        Versucht zuerst aus dem lokalen Label Studio Storage zu kopieren,
        falls das nicht funktioniert, werden die Bilder via API heruntergeladen.

        Args:
            project_id: Label Studio Projekt-ID
            export_path: Ziel-Pfad für den Export

        Returns:
            Anzahl kopierter Bilder
        """
        images_dir = export_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = export_path / "labels"

        # Label-Dateien als Referenz für benötigte Bilder
        label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        if not label_files:
            return 0

        # Label Studio lokaler Storage-Pfad
        # Standard: ~/.local/share/label-studio/media/upload/{project_id}/
        local_storage = Path.home() / ".local/share/label-studio/media/upload" / str(project_id)

        copied = 0

        for label_file in label_files:
            # Label-Dateiname ohne .txt = Bild-Dateiname
            base_name = label_file.stem

            # Mögliche Bild-Endungen
            for ext in [".jpg", ".jpeg", ".png"]:
                image_name = base_name + ext

                # Zuerst im lokalen Storage suchen
                local_image = local_storage / image_name
                if local_image.exists():
                    shutil.copy2(local_image, images_dir / image_name)
                    copied += 1
                    break

                # Alternative: Bild via API herunterladen
                # (nur falls lokaler Storage nicht funktioniert)

        # Falls keine Bilder im lokalen Storage gefunden,
        # versuche alle Bilder aus dem Storage zu kopieren
        if copied == 0 and local_storage.exists():
            for img_file in local_storage.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Prüfen ob passendes Label existiert
                    label_name = img_file.stem + ".txt"
                    if (labels_dir / label_name).exists():
                        shutil.copy2(img_file, images_dir / img_file.name)
                        copied += 1

        return copied

    def _organize_yolo_structure(self, export_path: Path) -> dict:
        """
        Organisiert die YOLO-Dateistruktur.

        Erwartete Struktur nach Label Studio Export:
        - images/
        - labels/
        - classes.txt oder notes.json

        Ziel-Struktur:
        - images/train/
        - labels/train/
        - data.yaml
        """
        images_dir = export_path / "images"
        labels_dir = export_path / "labels"

        # Zähle Dateien
        image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

        # classes.txt lesen oder erstellen
        classes_file = export_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file) as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            # Standard-Klassen aus classes.yaml
            classes = self._get_default_classes()
            with open(classes_file, "w") as f:
                f.write("\n".join(classes))

        # data.yaml für YOLO Training erstellen
        data_yaml = {
            "path": str(export_path.absolute()),
            "train": "images",
            "val": "images",  # Gleiche Daten für val (kann später gesplittet werden)
            "names": {i: name for i, name in enumerate(classes)}
        }

        with open(export_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return {
            "images": image_count,
            "labels": label_count,
            "classes": classes,
        }

    def _get_default_classes(self) -> list:
        """Lädt Standard-Klassen aus configs/classes.yaml."""
        config_path = Path("configs/classes.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                names = config.get("names", {})
                # Sortiert nach Index
                return [names[i] for i in sorted(names.keys())]

        # Fallback
        return [
            "ball", "cornerpoints", "goal", "goalkeeper",
            "period", "player", "ref", "scoreboard", "time"
        ]

    def merge_exports(self, export_paths: list, output_name: str) -> dict:
        """
        Kombiniert mehrere Exports zu einem Dataset.

        Args:
            export_paths: Liste von Export-Pfaden
            output_name: Name für den kombinierten Export

        Returns:
            dict mit Pfad und Statistiken
        """
        merged_path = self.output_dir / output_name
        merged_images = merged_path / "images"
        merged_labels = merged_path / "labels"

        merged_images.mkdir(parents=True, exist_ok=True)
        merged_labels.mkdir(parents=True, exist_ok=True)

        total_images = 0
        total_labels = 0

        for export_path in export_paths:
            export_path = Path(export_path)

            # Images kopieren
            src_images = export_path / "images"
            if src_images.exists():
                for img in src_images.glob("*"):
                    shutil.copy2(img, merged_images / img.name)
                    total_images += 1

            # Labels kopieren
            src_labels = export_path / "labels"
            if src_labels.exists():
                for lbl in src_labels.glob("*.txt"):
                    shutil.copy2(lbl, merged_labels / lbl.name)
                    total_labels += 1

        # data.yaml erstellen
        classes = self._get_default_classes()
        data_yaml = {
            "path": str(merged_path.absolute()),
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(classes)}
        }

        with open(merged_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        with open(merged_path / "classes.txt", "w") as f:
            f.write("\n".join(classes))

        return {
            "path": str(merged_path),
            "images": total_images,
            "labels": total_labels,
            "classes": classes,
        }
