import logging
from typing import Any, ClassVar

from llama_index.core.schema import MetadataMode, TransformComponent

from app.llm_classifier import classify_with_llama


logger = logging.getLogger(__name__)


class DocumentMetadataEnricher(TransformComponent):
    """
    Adds document classification metadata to each node.

    Strategy:
    1. Try rule-based classification first.
    2. Fall back to LLM classification.
    3. Validate output against allowed types.
    4. Fallback to 'generic' if anything fails.
    """

    DEFAULT_DOC_TYPE: ClassVar[str] = "generic"
    ALLOWED_DOC_TYPES: ClassVar[set[str]] = {"faq", "policy", "guide", "generic"}
    MAX_CLASSIFICATION_CHARS: ClassVar[int] = 2000

    def __call__(self, nodes: list[Any], **kwargs: Any) -> list[Any]:
        for node in nodes:
            metadata = node.metadata or {}
            file_name = metadata.get("file_name", "unknown")
            file_path = metadata.get("file_path", "")

            try:
                content = node.get_content(
                    metadata_mode=MetadataMode.NONE
                ).strip()

                doc_type = self._classify_document(content)
                metadata["doc_type"] = doc_type

                if file_path:
                    metadata.setdefault("source", file_path)

                node.metadata = metadata

                logger.info(
                    "Tagged document node",
                    extra={
                        "file_name": file_name,
                        "doc_type": doc_type,
                    },
                )

            except Exception:
                metadata["doc_type"] = self.DEFAULT_DOC_TYPE
                node.metadata = metadata

                logger.exception(
                    "Failed to enrich document node",
                    extra={"file_name": file_name},
                )

        return nodes

    def _classify_document(self, text: str) -> str:
        if not text:
            return self.DEFAULT_DOC_TYPE

        # Step 1: rule-based
        rule_based_type = self._classify_by_rules(text)
        if rule_based_type:
            return rule_based_type

        # Step 2: LLM fallback (with truncation)
        sample = text[: self.MAX_CLASSIFICATION_CHARS]
        llm_result = classify_with_llama(sample)

        if not llm_result or not isinstance(llm_result, str):
            return self.DEFAULT_DOC_TYPE

        normalized = llm_result.strip().lower()

        # ✅ Step 3: validation
        if normalized not in self.ALLOWED_DOC_TYPES:
            return self.DEFAULT_DOC_TYPE

        return normalized

    @staticmethod
    def _classify_by_rules(text: str) -> str | None:
        text_lower = text.lower()

        if "q:" in text_lower and "a:" in text_lower:
            return "faq"

        if "policy" in text_lower or "terms" in text_lower:
            return "policy"

        if "step 1" in text_lower or "guide" in text_lower:
            return "guide"

        return None
