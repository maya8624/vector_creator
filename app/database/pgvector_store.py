from urllib.parse import urlparse

from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings


class PgVectorStoreService:
    """
    Creates PGVectorStore using POSTGRES_URL.
    """

    def create_vector_store(self) -> PGVectorStore:
        parsed = urlparse(settings.POSTGRES_URL)

        return PGVectorStore.from_params(
            database=parsed.path.lstrip("/"),
            host=parsed.hostname,
            password=parsed.password,
            port=parsed.port,
            user=parsed.username,
            table_name=settings.VECTOR_TABLE,
            embed_dim=settings.EMBEDDING_DIM,
        )
