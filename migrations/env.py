"""
Alembic Environment Configuration

Dieses Modul konfiguriert Alembic für Datenbankmigrationen.
Es lädt die Datenbankverbindung aus .env und importiert alle Models.
"""
import os
import sys
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from alembic import context
from dotenv import load_dotenv

# .env laden
load_dotenv()

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Alembic Config Objekt
config = context.config

# Logging Setup
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Database URL aus Environment (direkt verwenden, nicht über config)
database_url = os.getenv("DATABASE_URL")

# Models importieren für Autogenerate
from src.db.models import Base
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generiert SQL ohne aktive Datenbankverbindung.
    """
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Verbindet zur Datenbank und führt Migrationen aus.
    """
    # Engine direkt erstellen statt über config (vermeidet % Interpolation Problem)
    connectable = create_engine(database_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
