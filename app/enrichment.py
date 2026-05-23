import json
import logging
import uuid
from pathlib import Path
from typing import Any, ClassVar

from llama_index.core.schema import MetadataMode, TransformComponent

from app.core.config import settings
from app.core.constants import (
    ALLOWED_DOC_TYPES,
    CLASSIFICATION_RULES,
    DOC_TYPE_GENERIC,
)
from app.llm_classifier import classify_with_llama


logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / \
    "data" / "doc_type_manifest.json"


def _load_manifest() -> dict[str, str]:
    if not _MANIFEST_PATH.exists():
        return {}
    with _MANIFEST_PATH.open() as f:
        return json.load(f)


class DocumentMetadataEnricher(TransformComponent):
    """
    Adds document classification metadata to each node.

    Strategy:
    1. Check manual manifest (filename → doc_type).
    2. Try rule-based classification.
    3. Fall back to LLM classification.
    4. Validate output against allowed types.
    5. Fallback to 'generic' if anything fails.
    """

    DEFAULT_DOC_TYPE: ClassVar[str] = DOC_TYPE_GENERIC
    ALLOWED_DOC_TYPES: ClassVar[frozenset[str]] = ALLOWED_DOC_TYPES
    MAX_CLASSIFICATION_CHARS: ClassVar[int] = 2000
    _manifest: ClassVar[dict[str, str]] = _load_manifest()

    def __call__(self, nodes: list[Any], **kwargs: Any) -> list[Any]:
        for node in nodes:
            metadata = node.metadata or {}
            file_name = metadata.get("file_name", "unknown")
            file_path = metadata.get("file_path", "")

            try:
                content = node.get_content(
                    metadata_mode=MetadataMode.NONE
                ).strip()

                doc_type = self._classify_document(content, file_name)
                metadata["doc_type"] = doc_type
                metadata["agency_id"] = settings.AGENCY_ID
                metadata["agency_name"] = settings.AGENCY_NAME
                metadata["user_id"] = settings.USER_ID
                metadata["doc_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{settings.AGENCY_ID}:{file_name}"))

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

    def _classify_document(self, text: str, filename: str = "") -> str:
        if not text:
            return self.DEFAULT_DOC_TYPE

        # Step 1: manifest lookup
        if filename and filename in self._manifest:
            return self._manifest[filename]

        # Step 2: rule-based
        rule_based_type = self._classify_by_rules(text)
        if rule_based_type:
            return rule_based_type

        # Step 3: LLM fallback (with truncation)
        sample = text[: self.MAX_CLASSIFICATION_CHARS]
        llm_result = classify_with_llama(sample, filename)

        if not llm_result or not isinstance(llm_result, str):
            return self.DEFAULT_DOC_TYPE

        normalized = llm_result.strip().lower()

        # Step 4: validation
        if normalized not in self.ALLOWED_DOC_TYPES:
            return self.DEFAULT_DOC_TYPE

        return normalized

    @staticmethod
    def _classify_by_rules(text: str) -> str | None:
        text_lower = text.lower()

        for doc_type, keywords, match_all in CLASSIFICATION_RULES:
            check = all if match_all else any
            if check(kw in text_lower for kw in keywords):
                return doc_type

        return None
