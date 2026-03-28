import asyncio
import logging

from app.embeddings.embedding_service import EmbeddingService
from app.rag.retriever import RagRetriever
from app.database.pgvector_store import PgVectorStoreService

logger = logging.getLogger(__name__)


async def main() -> None:

    retriever = RagRetriever(
        vector_store_service=PgVectorStoreService(),
        embedding_service=EmbeddingService(),
        similarity_top_k=3,
    )

    query = "What is the property address and bond amount for the contract?"
    results = await retriever.aretrieve(query)

    print(f"\nQuery: {query}\n")
    print(f"Retrieved {len(results)} result(s)\n")

    for index, result in enumerate(results, start=1):
        node = result.node
        metadata = node.metadata or {}

        print(f"--- Result {index} ---")
        print(f"Score: {result.score}")
        print(f"File: {metadata.get('file_name', 'unknown')}")
        print(f"Source: {metadata.get('source', 'unknown')}")
        print(f"Doc Type: {metadata.get('doc_type', 'unknown')}")
        print(f"Text Preview: {node.get_content()[:500]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
