"""
Label Studio API Client

Kommuniziert mit der Label Studio REST API.
"""
import os
import requests
from typing import Optional
from pathlib import Path


class LabelStudioError(Exception):
    """Fehler bei Label Studio Operationen."""
    pass


class LabelStudioClient:
    """Client für Label Studio API."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialisiert den Client.

        Args:
            url: Label Studio URL (default: aus .env)
            api_key: API Key (default: aus .env) - kann PAT (Refresh Token) oder Legacy Token sein
        """
        self.url = (url or os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")).rstrip("/")
        self.refresh_token = api_key or os.getenv("LABEL_STUDIO_API_KEY")
        self._access_token = None

        if not self.refresh_token:
            raise LabelStudioError(
                "LABEL_STUDIO_API_KEY nicht gesetzt. "
                "Bitte in .env konfigurieren oder als Parameter übergeben."
            )

        # Access Token holen (für JWT/PAT basierte Auth in Label Studio 1.20+)
        self._refresh_access_token()

    def _refresh_access_token(self):
        """Holt einen neuen Access Token vom Refresh Token (PAT)."""
        # Prüfen ob es ein JWT ist (PAT) oder ein Legacy Token
        if self.refresh_token.startswith("eyJ"):
            # JWT - muss zu Access Token umgewandelt werden
            try:
                response = requests.post(
                    f"{self.url}/api/token/refresh/",
                    json={"refresh": self.refresh_token},
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                self._access_token = data.get("access")
            except requests.exceptions.RequestException as e:
                raise LabelStudioError(f"Fehler beim Abrufen des Access Tokens: {e}")
        else:
            # Legacy Token - direkt verwenden
            self._access_token = self.refresh_token

    @property
    def headers(self):
        """Gibt die Auth-Header zurück."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Führt einen API Request aus."""
        url = f"{self.url}/api{endpoint}"

        # Content-Type für File-Uploads entfernen
        headers = self.headers.copy()
        if "files" in kwargs:
            del headers["Content-Type"]

        try:
            response = requests.request(method, url, headers=headers, **kwargs)

            # Bei 401 Token erneuern und nochmal versuchen
            if response.status_code == 401:
                self._refresh_access_token()
                headers = self.headers.copy()
                if "files" in kwargs:
                    del headers["Content-Type"]
                response = requests.request(method, url, headers=headers, **kwargs)

            response.raise_for_status()

            if response.content:
                return response.json()
            return {}

        except requests.exceptions.ConnectionError:
            raise LabelStudioError(
                f"Verbindung zu Label Studio ({self.url}) fehlgeschlagen. "
                "Ist Label Studio gestartet?"
            )
        except requests.exceptions.HTTPError as e:
            raise LabelStudioError(f"Label Studio API Fehler: {e}")

    def health_check(self) -> bool:
        """Prüft ob Label Studio erreichbar ist."""
        try:
            response = requests.get(f"{self.url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    # === Projekte ===

    def list_projects(self) -> list:
        """Listet alle Projekte."""
        return self._request("GET", "/projects/")

    def get_project(self, project_id: int) -> dict:
        """Holt ein Projekt."""
        return self._request("GET", f"/projects/{project_id}/")

    def create_project(
        self,
        title: str,
        description: str = "",
        label_config: Optional[str] = None
    ) -> dict:
        """
        Erstellt ein neues Projekt.

        Args:
            title: Projekt-Titel
            description: Beschreibung
            label_config: XML Label-Konfiguration

        Returns:
            Projekt-Daten
        """
        if not label_config:
            label_config = self._get_default_label_config()

        data = {
            "title": title,
            "description": description,
            "label_config": label_config,
        }

        return self._request("POST", "/projects/", json=data)

    def delete_project(self, project_id: int) -> None:
        """Löscht ein Projekt."""
        self._request("DELETE", f"/projects/{project_id}/")

    # === Tasks (Bilder zum Labeln) ===

    def import_tasks(self, project_id: int, tasks: list) -> dict:
        """
        Importiert Tasks (Bilder) in ein Projekt.

        Args:
            project_id: Projekt-ID
            tasks: Liste von Task-Daten [{"image": "url"}, ...]

        Returns:
            Import-Ergebnis
        """
        return self._request(
            "POST",
            f"/projects/{project_id}/import",
            json=tasks
        )

    def upload_image(self, project_id: int, image_path: str) -> dict:
        """
        Lädt ein Bild hoch und erstellt einen Task.

        Args:
            project_id: Projekt-ID
            image_path: Pfad zum Bild

        Returns:
            Upload-Ergebnis
        """
        path = Path(image_path)
        if not path.exists():
            raise LabelStudioError(f"Bild nicht gefunden: {image_path}")

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "image/jpeg")}
            return self._request(
                "POST",
                f"/projects/{project_id}/import",
                files=files
            )

    def upload_images_batch(self, project_id: int, image_paths: list) -> dict:
        """
        Lädt mehrere Bilder hoch.

        Args:
            project_id: Projekt-ID
            image_paths: Liste von Bildpfaden

        Returns:
            Upload-Ergebnis mit Anzahl erfolgreicher Uploads
        """
        results = {"success": 0, "failed": 0, "errors": []}

        for path in image_paths:
            try:
                self.upload_image(project_id, path)
                results["success"] += 1
            except LabelStudioError as e:
                results["failed"] += 1
                results["errors"].append(str(e))

        return results

    def list_tasks(self, project_id: int) -> list:
        """Listet alle Tasks eines Projekts."""
        result = self._request("GET", f"/projects/{project_id}/tasks/")
        return result if isinstance(result, list) else result.get("tasks", [])

    def get_task(self, task_id: int) -> dict:
        """Holt einen einzelnen Task."""
        return self._request("GET", f"/tasks/{task_id}/")

    # === Annotations ===

    def get_annotations(self, project_id: int) -> list:
        """
        Holt alle Annotations eines Projekts.

        Args:
            project_id: Projekt-ID

        Returns:
            Liste aller Annotations
        """
        tasks = self.list_tasks(project_id)
        annotations = []

        for task in tasks:
            if task.get("annotations"):
                annotations.extend(task["annotations"])

        return annotations

    def get_completed_tasks(self, project_id: int) -> list:
        """Holt nur Tasks die komplett annotiert wurden."""
        tasks = self.list_tasks(project_id)
        return [t for t in tasks if t.get("annotations")]

    # === Export ===

    def export_annotations(
        self,
        project_id: int,
        export_format: str = "YOLO"
    ) -> bytes:
        """
        Exportiert Annotations in einem bestimmten Format.

        Args:
            project_id: Projekt-ID
            export_format: Format (YOLO, COCO, JSON, etc.)

        Returns:
            Export-Daten als Bytes (ZIP für YOLO)
        """
        url = f"{self.url}/api/projects/{project_id}/export"
        params = {"exportType": export_format}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.content

    # === Hilfsmethoden ===

    def _get_default_label_config(self) -> str:
        """Gibt die Standard Label-Konfiguration zurück."""
        return """
        <View>
            <Image name="image" value="$image"/>
            <RectangleLabels name="label" toName="image">
                <Label value="player" background="#0000FF"/>
                <Label value="goalkeeper" background="#FFFF00"/>
                <Label value="ball" background="#FFA500"/>
                <Label value="ref" background="#808080"/>
                <Label value="goal" background="#FF0000"/>
                <Label value="cornerpoints" background="#00FF00"/>
                <Label value="scoreboard" background="#800080"/>
                <Label value="time" background="#008080"/>
                <Label value="period" background="#FFC0CB"/>
            </RectangleLabels>
        </View>
        """

    def get_project_url(self, project_id: int) -> str:
        """Gibt die URL zum Projekt im Browser zurück."""
        return f"{self.url}/projects/{project_id}/"

    def get_labeling_url(self, project_id: int) -> str:
        """Gibt die URL zum Labeling-Interface zurück."""
        return f"{self.url}/projects/{project_id}/data"

    def create_filtered_view(self, project_id: int, title: str, task_ids: list) -> dict:
        """
        Erstellt einen View/Tab mit Filter für bestimmte Task-IDs.

        Args:
            project_id: Projekt-ID
            title: Tab-Titel
            task_ids: Liste der Task-IDs die angezeigt werden sollen

        Returns:
            View-Daten inkl. ID
        """
        # Filter für Task-IDs: ID ist zwischen min und max
        min_id = min(task_ids)
        max_id = max(task_ids)

        view_data = {
            "project": project_id,
            "data": {
                "title": title,
                "type": "list",
                "target": "tasks",
                "filters": {
                    "conjunction": "and",
                    "items": [
                        {
                            "filter": "filter:tasks:id",
                            "operator": "greater_or_equal",
                            "type": "Number",
                            "value": min_id
                        },
                        {
                            "filter": "filter:tasks:id",
                            "operator": "less_or_equal",
                            "type": "Number",
                            "value": max_id
                        }
                    ]
                },
                "ordering": ["tasks:id"]
            }
        }

        return self._request("POST", "/dm/views/", json=view_data)

    def get_view_labeling_url(self, project_id: int, view_id: int) -> str:
        """Gibt die URL zum Labeling mit einem bestimmten View/Tab zurück."""
        return f"{self.url}/projects/{project_id}/data?tab={view_id}&labeling=1"
