import logging

from app.core.config import settings
from app.embeddings.embedding_service import EmbeddingService
from app.loaders.folder_loader import FolderLoader
from app.parsers.llama_parse_service import LlamaParseService
from app.pipelines.ingestion_pipeline import DocumentIngestionPipeline
from app.database.pgvector_service import PgVectorStoreService


logger = logging.getLogger(__name__)


def main() -> None:

    loader = FolderLoader(settings.DOCUMENT_PATH)
    parser_service = LlamaParseService()
    embedding_service = EmbeddingService()
    vector_store_service = PgVectorStoreService()

    ingestion_pipeline = DocumentIngestionPipeline(
        parser_service=parser_service,
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
    )

    files = loader.load_files()

    if not files:
        logger.warning(
            "No files found in document path: %s",
            settings.DOCUMENT_PATH
        )
        return

    logger.info("Found %d files to ingest", len(files))

    for file_path in files:
        try:
            ingestion_pipeline.run(file_path)
            logger.info("Successfully ingested file: %s", file_path)
        except Exception:
            logger.exception("Failed to ingest file: %s", file_path)


if __name__ == "__main__":
    main()
