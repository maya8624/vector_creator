import os
import logging
from pathlib import Path
from app.core.config import settings
from ingestion import IngestionService
from pathlib import Path
from app.parsers.llama_parse_service import LlamaParseService

parser = LlamaParseService()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    # Ensure data directory exists
    doc_path = Path(settings.DOCUMENT_PATH)

    if not doc_path.exists():
        doc_path.mkdir(parents=True, exist_ok=True)
        return f"Folder created at {doc_path}. Please add files to ingest."

    parser = LlamaParseService()
    docs = parser.parse(Path("./data/documents/market_report_q1_2025.pdf"))

    print(len(docs))
    print(docs[0].text[:500])

    # logger.info("Starting ingestion from '%s' into '%s'",
    #             doc_path, settings.CHROMA_DB_PATH)

    # service = IngestionService(db_path=settings.CHROMA_DB_PATH)

    # service.run(doc_path=doc_path)


if __name__ == "__main__":
    main()
