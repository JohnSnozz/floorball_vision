"""Initial database schema

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01

Erstellt alle Basis-Tabellen für das Floorball Vision Projekt:
- videos: Hochgeladene/heruntergeladene Videos
- calibrations: Kamera-Kalibrierungen
- labeling_projects: Label Studio Projekte
- training_runs: YOLO Trainingsläufe
- active_model: Aktuell aktives Modell
- analysis_jobs: Analyse-Aufträge
- player_positions: Frame-für-Frame Positionen
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Videos Tabelle
    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(255)),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("duration", sa.Float),
        sa.Column("fps", sa.Float),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("codec", sa.String(50)),
        sa.Column("source_type", sa.String(50)),
        sa.Column("source_url", sa.String(1000)),
        sa.Column("status", sa.String(50), default="uploaded"),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Calibrations Tabelle
    op.create_table(
        "calibrations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("name", sa.String(255)),
        sa.Column("calibration_data", postgresql.JSON, nullable=False),
        sa.Column("fisheye_params", postgresql.JSON),
        sa.Column("field_length", sa.Float, default=40.0),
        sa.Column("field_width", sa.Float, default=20.0),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Labeling Projects Tabelle
    op.create_table(
        "labeling_projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("label_studio_id", sa.Integer, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("label_config", sa.Text),
        sa.Column("classes", postgresql.JSON),
        sa.Column("images_path", sa.String(500)),
        sa.Column("total_images", sa.Integer, default=0),
        sa.Column("labeled_images", sa.Integer, default=0),
        sa.Column("auto_train_enabled", sa.Boolean, default=False),
        sa.Column("min_labels_for_training", sa.Integer, default=100),
        sa.Column("status", sa.String(50), default="active"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Training Runs Tabelle
    op.create_table(
        "training_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("labeling_project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("labeling_projects.id")),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("base_model", sa.String(255), default="yolov8n.pt"),
        sa.Column("hyperparameters", postgresql.JSON),
        sa.Column("train_images", sa.Integer),
        sa.Column("val_images", sa.Integer),
        sa.Column("final_map50", sa.Float),
        sa.Column("final_map50_95", sa.Float),
        sa.Column("training_time", sa.Float),
        sa.Column("model_path", sa.String(500)),
        sa.Column("results_path", sa.String(500)),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Active Model Tabelle
    op.create_table(
        "active_model",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("training_runs.id")),
        sa.Column("model_path", sa.String(500), nullable=False),
        sa.Column("model_name", sa.String(255)),
        sa.Column("map50", sa.Float),
        sa.Column("activated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Analysis Jobs Tabelle
    op.create_table(
        "analysis_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("calibration_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("calibrations.id")),
        sa.Column("job_type", sa.String(50), default="full"),
        sa.Column("start_time", sa.Float),
        sa.Column("end_time", sa.Float),
        sa.Column("sample_rate", sa.Integer, default=1),
        sa.Column("total_frames", sa.Integer),
        sa.Column("processed_frames", sa.Integer, default=0),
        sa.Column("progress_percent", sa.Float, default=0.0),
        sa.Column("detected_players", sa.Integer),
        sa.Column("detected_balls", sa.Integer),
        sa.Column("celery_task_id", sa.String(255)),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Player Positions Tabelle
    op.create_table(
        "player_positions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("analysis_job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("analysis_jobs.id"), nullable=False),
        sa.Column("frame_number", sa.Integer, nullable=False),
        sa.Column("timestamp", sa.Float, nullable=False),
        sa.Column("object_type", sa.String(50), nullable=False),
        sa.Column("track_id", sa.Integer),
        sa.Column("team_id", sa.Integer),
        sa.Column("bbox_x1", sa.Float),
        sa.Column("bbox_y1", sa.Float),
        sa.Column("bbox_x2", sa.Float),
        sa.Column("bbox_y2", sa.Float),
        sa.Column("confidence", sa.Float),
        sa.Column("field_x", sa.Float),
        sa.Column("field_y", sa.Float),
        sa.Column("extra_data", postgresql.JSON),
    )

    # Indices
    op.create_index(
        "ix_player_positions_job_frame",
        "player_positions",
        ["analysis_job_id", "frame_number"]
    )
    op.create_index(
        "ix_player_positions_job_track",
        "player_positions",
        ["analysis_job_id", "track_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_player_positions_job_track", table_name="player_positions")
    op.drop_index("ix_player_positions_job_frame", table_name="player_positions")
    op.drop_table("player_positions")
    op.drop_table("analysis_jobs")
    op.drop_table("active_model")
    op.drop_table("training_runs")
    op.drop_table("labeling_projects")
    op.drop_table("calibrations")
    op.drop_table("videos")
