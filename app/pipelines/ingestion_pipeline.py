
import logging

from pathlib import Path

from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.embeddings.embedding_service import EmbeddingService
from app.parsers.llama_parse_service import LlamaParseService
from app.database.pgvector_store import PgVectorStoreService
from app.enrichment import DocumentMetadataEnricher

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Orchestrates the document ingestion workflow:
    parse -> enrich -> split -> embed -> store
    """

    def __init__(
        self,
        parser_service: LlamaParseService,
        embedding_service: EmbeddingService,
        vector_store_service: PgVectorStoreService,
    ) -> None:
        self._parser_service = parser_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store_service.create_vector_store()

        self._pipeline = IngestionPipeline(
            transformations=[
                DocumentMetadataEnricher(),
                MarkdownNodeParser(),
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                self._embedding_service.model,
            ],
            vector_store=self._vector_store,
            docstore=SimpleDocumentStore(),
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

    def run(self, file_path: Path) -> None:
        """
        Ingest a single file into the vector store.
        """
        logger.info("Starting ingestion for file: %s", file_path)

        documents = self._parser_service.parse(file_path)

        if not documents:
            logger.warning("No documents found for file: %s", file_path)
            return

        self._pipeline.run(documents=documents)

        logger.info("Completed ingestion for file: %s", file_path)
