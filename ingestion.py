import logging
import chromadb
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import DocstoreStrategy
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from enrichment import DocumentMetadataEnricher
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocIngestionPipeline")


class IngestionService:
    def __init__(self, db_path: str):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL)

        self.db = chromadb.PersistentClient(path=db_path)

        self.chroma_collection = self.db.get_or_create_collection(
            "real-estate-brain")

        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection)

    def run(self, doc_path: str):
        try:
            parser = LlamaParse(api_key=settings.LLAMA_CLOUD_API_KEY,
                                result_type="markdown")

            file_extractor = {".pdf": parser,
                              ".docx": parser,
                              ".csv": parser,
                              ".txt": parser}

            pipeline = IngestionPipeline(
                transformations=[
                    DocumentMetadataEnricher(),
                    MarkdownNodeParser(),
                    Settings.embed_model,
                ],

                vector_store=self.vector_store,
                docstore=SimpleDocumentStore(),
                docstore_strategy=DocstoreStrategy.UPSERTS,
            )

            documents = SimpleDirectoryReader(
                input_dir=doc_path,
                file_extractor=file_extractor,
                recursive=True
            ).load_data()

            pipeline.run(documents=documents)
            logger.info("✅ Vector Database created/updated successfully.")

        except Exception as e:
            logger.error("🛑 Ingestion crashed: %s", e, exc_info=True)
