"""
SQLAlchemy Database Models

Alle Datenbank-Tabellen für das Floorball Vision Projekt.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy Base Model"""
    pass


class Video(Base):
    """
    Videos Tabelle

    Speichert hochgeladene oder heruntergeladene Videos.
    """
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Metadaten
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(String(500), nullable=False)

    # Video-Eigenschaften
    duration = Column(Float)  # Sekunden
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    codec = Column(String(50))

    # Quelle
    source_type = Column(String(50))  # 'upload', 'youtube', 'local'
    source_url = Column(String(1000))  # YouTube URL falls relevant

    # Status
    status = Column(String(50), default="uploaded")  # uploaded, processing, ready, error
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    calibrations = relationship("Calibration", back_populates="video")
    analysis_jobs = relationship("AnalysisJob", back_populates="video")


class Calibration(Base):
    """
    Kamera-Kalibrierungen

    Speichert Homographie-Matrizen für die Koordinaten-Transformation.
    """
    __tablename__ = "calibrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)

    # Kalibrierungsname
    name = Column(String(255))

    # Kalibrierungsdaten (JSON)
    # Enthält: source_points, target_points, homography_matrix
    calibration_data = Column(JSON, nullable=False)

    # Optionale Fisheye-Korrektur Parameter
    fisheye_params = Column(JSON)  # Distortion coefficients

    # Felddimensionen (für Referenz)
    field_length = Column(Float, default=40.0)  # Meter
    field_width = Column(Float, default=20.0)   # Meter

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="calibrations")
    analysis_jobs = relationship("AnalysisJob", back_populates="calibration")


class LabelingProject(Base):
    """
    Label Studio Projekte

    Verknüpft lokale Daten mit Label Studio Projekten.
    """
    __tablename__ = "labeling_projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Label Studio Referenz
    label_studio_id = Column(Integer, unique=True)  # ID in Label Studio
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Konfiguration
    label_config = Column(Text)  # XML Label Config
    classes = Column(JSON)  # Liste der Klassen ['player', 'ball', 'goalkeeper']

    # Pfade
    images_path = Column(String(500))  # Lokaler Pfad zu Bildern

    # Statistiken
    total_images = Column(Integer, default=0)
    labeled_images = Column(Integer, default=0)

    # Webhook für Auto-Training
    auto_train_enabled = Column(Boolean, default=False)
    min_labels_for_training = Column(Integer, default=100)

    # Status
    status = Column(String(50), default="active")  # active, archived

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_runs = relationship("TrainingRun", back_populates="labeling_project")


class TrainingRun(Base):
    """
    YOLO Training Runs

    Protokolliert alle Trainingsläufe.
    """
    __tablename__ = "training_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    labeling_project_id = Column(UUID(as_uuid=True), ForeignKey("labeling_projects.id"))

    # Training-Konfiguration
    model_name = Column(String(255), nullable=False)
    base_model = Column(String(255), default="yolov8n.pt")

    # Hyperparameter (JSON)
    hyperparameters = Column(JSON)  # epochs, batch_size, imgsz, etc.

    # Dataset Info
    train_images = Column(Integer)
    val_images = Column(Integer)

    # Ergebnisse
    final_map50 = Column(Float)  # mAP@0.5
    final_map50_95 = Column(Float)  # mAP@0.5:0.95
    training_time = Column(Float)  # Sekunden

    # Pfade
    model_path = Column(String(500))  # Pfad zum trainierten Modell
    results_path = Column(String(500))  # Pfad zu Trainingsresultaten

    # Status
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    error_message = Column(Text)

    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    labeling_project = relationship("LabelingProject", back_populates="training_runs")


class ActiveModel(Base):
    """
    Aktives Modell

    Zeigt auf das aktuell aktive Modell für Analyse.
    Nur ein Modell kann gleichzeitig aktiv sein.
    """
    __tablename__ = "active_model"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_run_id = Column(UUID(as_uuid=True), ForeignKey("training_runs.id"))

    # Modell-Info
    model_path = Column(String(500), nullable=False)
    model_name = Column(String(255))

    # Performance
    map50 = Column(Float)

    # Timestamps
    activated_at = Column(DateTime, default=datetime.utcnow)


class AnalysisJob(Base):
    """
    Analyse Jobs

    Ein Job analysiert ein Video (komplett oder Snippet).
    """
    __tablename__ = "analysis_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    calibration_id = Column(UUID(as_uuid=True), ForeignKey("calibrations.id"))

    # Job-Typ
    job_type = Column(String(50), default="full")  # 'preview', 'full'

    # Zeitbereich (für Snippets)
    start_time = Column(Float)  # Sekunden
    end_time = Column(Float)    # Sekunden

    # Konfiguration
    sample_rate = Column(Integer, default=1)  # Jedes n-te Frame

    # Progress
    total_frames = Column(Integer)
    processed_frames = Column(Integer, default=0)
    progress_percent = Column(Float, default=0.0)

    # Ergebnisse (Zusammenfassung)
    detected_players = Column(Integer)
    detected_balls = Column(Integer)

    # Celery Task ID
    celery_task_id = Column(String(255))

    # Status
    status = Column(String(50), default="pending")  # pending, running, paused, completed, failed
    error_message = Column(Text)

    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="analysis_jobs")
    calibration = relationship("Calibration", back_populates="analysis_jobs")
    player_positions = relationship("PlayerPosition", back_populates="analysis_job")


class PlayerPosition(Base):
    """
    Spieler-Positionen

    Speichert Frame-für-Frame Positionen aller erkannten Objekte.
    Partitioniert nach job_id für effiziente Abfragen.
    """
    __tablename__ = "player_positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id"), nullable=False)

    # Frame-Info
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # Sekunden im Video

    # Objekt-Info
    object_type = Column(String(50), nullable=False)  # 'player', 'ball', 'goalkeeper', 'referee'
    track_id = Column(Integer)  # ByteTrack ID
    team_id = Column(Integer)  # 0 = Team A, 1 = Team B, -1 = unbekannt

    # Bild-Koordinaten (Pixel)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    confidence = Column(Float)

    # Feld-Koordinaten (Meter, nach Homographie)
    field_x = Column(Float)
    field_y = Column(Float)

    # Zusätzliche Daten (optional)
    extra_data = Column(JSON)  # Für zukünftige Erweiterungen

    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="player_positions")

    # Indices für schnelle Abfragen
    __table_args__ = (
        Index("ix_player_positions_job_frame", "analysis_job_id", "frame_number"),
        Index("ix_player_positions_job_track", "analysis_job_id", "track_id"),
    )
