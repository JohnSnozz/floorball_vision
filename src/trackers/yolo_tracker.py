"""
YOLO Tracker - Spieler-Detection und Tracking mit ByteTrack/BoT-SORT.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """
    Prüft ob ein Punkt innerhalb eines Polygons liegt (Ray-Casting).

    Args:
        x: X-Koordinate des Punkts
        y: Y-Koordinate des Punkts
        polygon: Liste von [x, y] Punkten

    Returns:
        True wenn Punkt im Polygon
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class YOLOTracker:
    """
    YOLO-basierter Tracker für Spieler-Detection und Tracking.

    Unterstützt ByteTrack, BoT-SORT und BoT-SORT mit Re-ID.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_config: Optional[str] = None,
        conf_threshold: float = 0.3
    ):
        """
        Initialisiert den Tracker.

        Args:
            model_path: Pfad zum YOLO-Modell (.pt)
            tracker_config: Pfad zur Tracker-Konfiguration (.yaml)
            conf_threshold: Konfidenz-Schwellwert für Detections
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold

    def track_video(
        self,
        video_path: str,
        target_fps: float = 2.0,
        field_polygon: Optional[List[List[float]]] = None,
        on_frame: Optional[Callable[[int, int, List[Dict]], None]] = None,
        team_assigner: Optional[Any] = None,
        jersey_detector: Optional[Any] = None,
        team_assign_interval: float = 10.0,
        jersey_ocr_interval: float = 5.0
    ) -> Dict[str, Any]:
        """
        Führt Tracking über ein Video aus.

        Args:
            video_path: Pfad zum Video
            target_fps: Ziel-FPS für Tracking (reduziert Frames)
            field_polygon: Spielfeld-Polygon für Filterung
            on_frame: Callback(frame_idx, total_frames, detections) für Progress
            team_assigner: TeamAssigner-Instanz für Team-Zuweisung
            jersey_detector: JerseyOCR-Instanz für Rückennummern
            team_assign_interval: Sekunden zwischen Team-Assignments pro Track
            jersey_ocr_interval: Sekunden zwischen OCR-Versuchen pro Track

        Returns:
            Dict mit Tracking-Ergebnissen
        """
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame-Skip berechnen
        frame_skip = max(1, int(video_fps / target_fps))
        actual_fps = video_fps / frame_skip

        # Voting-Strukturen für stabile Zuweisungen
        track_team_votes: Dict[int, Dict[str, int]] = {}
        track_number_votes: Dict[int, Dict[int, List[Dict]]] = {}

        # Zeit-basiertes Tracking
        track_last_team_assign: Dict[int, float] = {}
        track_last_jersey_ocr: Dict[int, float] = {}

        results = {
            "video_fps": video_fps,
            "tracking_fps": actual_fps,
            "frame_skip": frame_skip,
            "total_frames": total_frames,
            "tracked_frames": 0,
            "model": self.model_path,
            "conf_threshold": self.conf_threshold,
            "field_polygon": field_polygon,
            "frames": []
        }

        frame_idx = 0
        tracked_count = 0
        filtered_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Nur jeden n-ten Frame tracken
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            current_timestamp = frame_idx / video_fps

            # Progress Callback
            if on_frame:
                on_frame(tracked_count, total_frames // frame_skip, [])

            # YOLO Tracking
            track_kwargs = {
                "persist": True,
                "conf": self.conf_threshold,
                "verbose": False
            }
            if self.tracker_config:
                track_kwargs["tracker"] = self.tracker_config

            yolo_results = self.model.track(frame, **track_kwargs)

            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": current_timestamp,
                "detections": []
            }

            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes[i]
                        bbox = box.xyxy[0].tolist()

                        # Fuss-Position für Spielfeld-Check
                        foot_x = (bbox[0] + bbox[2]) / 2
                        foot_y = bbox[3]

                        in_field = True
                        if field_polygon:
                            in_field = point_in_polygon(foot_x, foot_y, field_polygon)
                            if not in_field:
                                filtered_count += 1

                        detection = {
                            "bbox": bbox,
                            "conf": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "class_name": self.model.names[int(box.cls[0])],
                            "in_field": in_field
                        }

                        # Track-ID
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0])
                            detection["track_id"] = track_id

                        # Team-Zuweisung (zeitbasiert)
                        if in_field and track_id is not None and team_assigner:
                            last_assign = track_last_team_assign.get(track_id, -float('inf'))
                            if (current_timestamp - last_assign) >= team_assign_interval:
                                try:
                                    team, team_conf = team_assigner.assign_team(frame, bbox)
                                    detection["team"] = team
                                    detection["team_conf"] = team_conf
                                    track_last_team_assign[track_id] = current_timestamp

                                    # Votes akkumulieren
                                    if track_id not in track_team_votes:
                                        track_team_votes[track_id] = {"team1": 0, "team2": 0, "referee": 0}
                                    if team in track_team_votes[track_id]:
                                        track_team_votes[track_id][team] += 1
                                except Exception as e:
                                    print(f"Team assignment error: {e}")

                        # Jersey-Nummer (zeitbasiert)
                        if in_field and track_id is not None and jersey_detector:
                            last_ocr = track_last_jersey_ocr.get(track_id, -float('inf'))
                            if (current_timestamp - last_ocr) >= jersey_ocr_interval:
                                try:
                                    jersey_num, num_conf = jersey_detector.detect(frame, bbox)
                                    track_last_jersey_ocr[track_id] = current_timestamp
                                    if jersey_num is not None and num_conf > 0.3:
                                        detection["jersey_number"] = jersey_num
                                        detection["jersey_conf"] = num_conf

                                        # Number votes akkumulieren
                                        if track_id not in track_number_votes:
                                            track_number_votes[track_id] = {}
                                        if jersey_num not in track_number_votes[track_id]:
                                            track_number_votes[track_id][jersey_num] = []
                                        track_number_votes[track_id][jersey_num].append({
                                            "conf": num_conf,
                                            "frame_idx": frame_idx
                                        })
                                except Exception as e:
                                    print(f"Jersey detection error: {e}")

                        frame_data["detections"].append(detection)

            results["frames"].append(frame_data)
            tracked_count += 1
            frame_idx += 1

        cap.release()

        results["tracked_frames"] = tracked_count
        results["filtered_count"] = filtered_count

        # Finale Team-Zuweisungen
        results["track_team_assignments"] = self._compute_final_team_assignments(track_team_votes)

        # Finale Jersey-Nummern
        results["track_jersey_numbers"] = self._compute_final_jersey_numbers(track_number_votes)

        return results

    def _compute_final_team_assignments(
        self,
        track_team_votes: Dict[int, Dict[str, int]]
    ) -> Dict[str, Dict[str, Any]]:
        """Berechnet finale Team-Zuweisungen basierend auf Voting."""
        final_assignments = {}
        for track_id, votes in track_team_votes.items():
            best_team = max(votes.keys(), key=lambda t: votes[t])
            total_votes = sum(votes.values())
            confidence = votes[best_team] / total_votes if total_votes > 0 else 0
            final_assignments[str(track_id)] = {
                "team": best_team,
                "confidence": confidence,
                "votes": votes
            }
        return final_assignments

    def _compute_final_jersey_numbers(
        self,
        track_number_votes: Dict[int, Dict[int, List[Dict]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Berechnet finale Jersey-Nummern basierend auf Voting."""
        final_numbers = {}
        for track_id, number_votes in track_number_votes.items():
            if not number_votes:
                continue

            # Gewichtete Scores
            number_scores = {}
            for num, sightings in number_votes.items():
                weighted_score = sum(s["conf"] for s in sightings)
                avg_conf = weighted_score / len(sightings) if sightings else 0
                number_scores[num] = {
                    "score": weighted_score,
                    "count": len(sightings),
                    "avg_conf": avg_conf
                }

            # Top-Kandidaten
            sorted_numbers = sorted(
                number_scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )

            candidates = []
            for num, data in sorted_numbers[:3]:
                candidates.append({
                    "number": num,
                    "count": data["count"],
                    "avg_confidence": round(data["avg_conf"], 3)
                })

            if candidates:
                final_numbers[str(track_id)] = {
                    "primary": candidates[0]["number"],
                    "primary_confidence": candidates[0]["avg_confidence"],
                    "candidates": candidates
                }

        return final_numbers

    def track_video_segment(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int = -1,
        frame_step: int = 1,
        field_polygon: Optional[List[List[float]]] = None,
        team_assigner: Optional[Any] = None,
        jersey_detector: Optional[Any] = None,
        team_assign_interval: float = 10.0,
        jersey_ocr_interval: float = 5.0
    ) -> Dict[str, Any]:
        """
        Führt Tracking über ein Video-Segment aus.

        Args:
            video_path: Pfad zum Video
            start_frame: Start-Frame (inklusive)
            end_frame: End-Frame (exklusive), -1 = bis Ende
            frame_step: Nur jeden n-ten Frame verarbeiten
            field_polygon: Spielfeld-Polygon für Filterung
            team_assigner: TeamAssigner-Instanz für Team-Zuweisung
            jersey_detector: JerseyOCR-Instanz für Rückennummern
            team_assign_interval: Sekunden zwischen Team-Assignments pro Track
            jersey_ocr_interval: Sekunden zwischen OCR-Versuchen pro Track

        Returns:
            Dict mit Tracking-Ergebnissen
        """
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame < 0:
            end_frame = total_frames

        # Zum Start-Frame springen
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Voting-Strukturen für stabile Zuweisungen
        track_team_votes: Dict[int, Dict[str, int]] = {}
        track_number_votes: Dict[int, Dict[int, List[Dict]]] = {}

        # Zeit-basiertes Tracking für Team/Jersey
        track_last_team_assign: Dict[int, float] = {}
        track_last_jersey_ocr: Dict[int, float] = {}

        results = {
            "video_fps": video_fps,
            "frame_step": frame_step,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "model": self.model_path,
            "conf_threshold": self.conf_threshold,
            "frames": []
        }

        frame_idx = start_frame
        tracked_count = 0

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Nur jeden n-ten Frame tracken
            if (frame_idx - start_frame) % frame_step != 0:
                frame_idx += 1
                continue

            current_timestamp = frame_idx / video_fps

            # YOLO Tracking
            track_kwargs = {
                "persist": True,
                "verbose": False,
                "conf": self.conf_threshold
            }

            if self.tracker_config:
                track_kwargs["tracker"] = self.tracker_config

            yolo_results = self.model.track(frame, **track_kwargs)

            # Detektionen extrahieren
            frame_detections = []
            if yolo_results and yolo_results[0].boxes is not None:
                boxes = yolo_results[0].boxes

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())

                    class_name = self.model.names.get(cls_id, f"class_{cls_id}")

                    # Center-Point berechnen
                    cx = (x1 + x2) / 2
                    cy_foot = y2  # Fuß-Position

                    # Polygon-Filter
                    in_field = True
                    if field_polygon:
                        in_field = point_in_polygon(cx, cy_foot, field_polygon)
                        if not in_field:
                            continue

                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    detection = {
                        "bbox": bbox,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "track_id": track_id,
                        "center": [float(cx), float(cy_foot)],
                        "in_field": in_field
                    }

                    # Team-Zuweisung (zeitbasiert)
                    if in_field and track_id is not None and team_assigner:
                        last_assign = track_last_team_assign.get(track_id, -float('inf'))
                        if (current_timestamp - last_assign) >= team_assign_interval:
                            try:
                                team, team_conf = team_assigner.assign_team(frame, bbox)
                                detection["team"] = team
                                detection["team_conf"] = team_conf
                                track_last_team_assign[track_id] = current_timestamp

                                # Votes akkumulieren
                                if track_id not in track_team_votes:
                                    track_team_votes[track_id] = {"team1": 0, "team2": 0, "referee": 0}
                                if team in track_team_votes[track_id]:
                                    track_team_votes[track_id][team] += 1
                            except Exception as e:
                                print(f"Team assignment error: {e}")

                    # Jersey-Nummer (zeitbasiert)
                    if in_field and track_id is not None and jersey_detector:
                        last_ocr = track_last_jersey_ocr.get(track_id, -float('inf'))
                        if (current_timestamp - last_ocr) >= jersey_ocr_interval:
                            try:
                                jersey_num, num_conf = jersey_detector.detect(frame, bbox)
                                track_last_jersey_ocr[track_id] = current_timestamp
                                if jersey_num is not None and num_conf > 0.3:
                                    detection["jersey_number"] = jersey_num
                                    detection["jersey_conf"] = num_conf

                                    # Number votes akkumulieren
                                    if track_id not in track_number_votes:
                                        track_number_votes[track_id] = {}
                                    if jersey_num not in track_number_votes[track_id]:
                                        track_number_votes[track_id][jersey_num] = []
                                    track_number_votes[track_id][jersey_num].append({
                                        "conf": num_conf,
                                        "frame_idx": frame_idx
                                    })
                            except Exception as e:
                                print(f"Jersey detection error: {e}")

                    frame_detections.append(detection)

            results["frames"].append({
                "frame_idx": frame_idx,
                "timestamp": current_timestamp,
                "detections": frame_detections
            })

            tracked_count += 1
            frame_idx += 1

        cap.release()

        results["tracked_frames"] = tracked_count

        # Finale Team-Zuweisungen
        results["track_team_assignments"] = self._compute_final_team_assignments(track_team_votes)

        # Finale Jersey-Nummern
        results["track_jersey_numbers"] = self._compute_final_jersey_numbers(track_number_votes)

        return results
