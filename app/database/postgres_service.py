from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings


class PostgresService:
    """
    Responsible for PostgreSQL engine and session creation.
    """

    def __init__(self) -> None:
        self._connection_string = settings.POSTGRES_URL

        self._engine = create_engine(
            self._connection_string,
            pool_pre_ping=True,
        )

        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    def create_session(self) -> Session:
        return self._session_factory()
