import logging
from llama_index.core.schema import TransformComponent, MetadataMode
from app.llm_classifier import classify_with_llama

logger = logging.getLogger(__name__)


class DocumentMetadataEnricher(TransformComponent):
    """
    SOLID: Single Responsibility.
    Handles document classification and metadata tagging.
    """

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            try:
                # Get raw text to classify
                content = node.get_content(metadata_mode=MetadataMode.NONE)
                doc_type = self._classify(content)

                # Tag the node
                node.metadata["doc_type"] = doc_type
                logger.info(
                    f"Tagged {node.metadata.get('file_name')} as {doc_type}")
            except Exception as e:
                logger.error(
                    f"Enrichment failed for {node.metadata.get('file_name')}: {e}")
                node.metadata["doc_type"] = "generic"
        return nodes

    def _classify(self, text: str) -> str:
        """Your specific rule-based logic + Ollama fallback."""
        text_lower = text.lower()
        if "q:" in text_lower and "a:" in text_lower:
            return "faq"
        if "policy" in text_lower or "terms" in text_lower:
            return "policy"
        if "step 1" in text_lower or "guide" in text_lower:
            return "guide"

        return classify_with_llama(text)
