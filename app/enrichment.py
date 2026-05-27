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
    "data" / "doc_metadata_manifest.json"

_MANIFEST_EXTRA_FIELDS: frozenset[str] = frozenset({
    "lease_start",
    "lease_end",
    "billing_period",
    "inspection_date",
    "offer_expiry",
    "bond_reference",
    "statement_period",
    "notice_type",
    "effective_date",
})


def _load_manifest() -> dict[str, dict]:
    if not _MANIFEST_PATH.exists():
        return {}
    with _MANIFEST_PATH.open() as f:
        data = json.load(f)
    return {doc["filename"]: doc for doc in data.get("documents", [])}


class DocumentMetadataEnricher(TransformComponent):
    """
    Adds document classification metadata to each node.

    Strategy:
    1. Check manual manifest (filename → doc entry) — resolves doc_type and all
       structured fields (tenant_id, property_id, description, doc-type-specific).
    2. Try rule-based classification.
    3. Fall back to LLM classification.
    4. Validate output against allowed types.
    5. Default to 'generic' if all strategies fail.
    """

    DEFAULT_DOC_TYPE: ClassVar[str] = DOC_TYPE_GENERIC
    ALLOWED_DOC_TYPES: ClassVar[frozenset[str]] = ALLOWED_DOC_TYPES
    MAX_CLASSIFICATION_CHARS: ClassVar[int] = 2000
    _manifest: ClassVar[dict[str, dict]] = _load_manifest()

    def __call__(self, nodes: list[Any], **kwargs: Any) -> list[Any]:
        for node in nodes:
            metadata = node.metadata or {}
            file_name = metadata.get("file_name", "unknown")
            file_path = metadata.get("file_path", "")

            try:
                content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
                manifest_entry = self._manifest.get(file_name, {})

                metadata["doc_type"] = self._classify_document(content, file_name, manifest_entry)
                metadata["agency_id"] = settings.AGENCY_ID
                metadata["agency_name"] = settings.AGENCY_NAME
                metadata["tenant_id"] = manifest_entry.get("tenant_id")
                metadata["property_id"] = manifest_entry.get("property_id")
                metadata["description"] = manifest_entry.get("description")
                metadata["doc_id"] = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS, f"{settings.AGENCY_ID}:{file_name}"
                ))

                for field in _MANIFEST_EXTRA_FIELDS:
                    if field in manifest_entry:
                        metadata[field] = manifest_entry[field]

                if file_path:
                    metadata.setdefault("source", file_path)

                node.metadata = metadata

                logger.info(
                    "Tagged document node",
                    extra={
                        "file_name": file_name,
                        "doc_type": metadata["doc_type"],
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

    def _classify_document(
        self,
        text: str,
        filename: str = "",
        manifest_entry: dict | None = None,
    ) -> str:
        if not text:
            return self.DEFAULT_DOC_TYPE

        # Step 1: manifest lookup
        if manifest_entry and "document_type" in manifest_entry:
            return manifest_entry["document_type"]

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
