from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from app.embeddings.embedding_service import EmbeddingService
from app.database.pgvector_service import PgVectorStoreService


class RagRetriever:
    """
    Thin retrieval layer over the existing pgvector-backed LlamaIndex store.
    """

    def __init__(
        self,
        vector_store_service: PgVectorStoreService,
        embedding_service: EmbeddingService,
        similarity_top_k: int = 5,
        mmr_threshold: float = 0.7,
    ) -> None:
        self._vector_store = vector_store_service.create_vector_store()
        self._embed_model = embedding_service.model
        self._similarity_top_k = similarity_top_k
        self._mmr_threshold = mmr_threshold

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )

    def _build_retriever(self):
        return self._index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            vector_store_query_mode=VectorStoreQueryMode.MMR,
            mmr_threshold=self._mmr_threshold,
        )

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        retriever = self._build_retriever()
        return await retriever.aretrieve(query)

    # def retrieve(self, query: str) -> list[NodeWithScore]:
    #     if not query.strip():
    #         raise ValueError("Query cannot be empty.")

    #     retriever = self._build_retriever()
    #     return retriever.retrieve(query)
