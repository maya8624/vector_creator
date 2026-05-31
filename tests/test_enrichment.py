import types
from unittest.mock import MagicMock, patch

import pytest

from app.core.constants import (
    DOC_TYPE_BOND_LODGEMENT,
    DOC_TYPE_FAQ,
    DOC_TYPE_GENERIC,
    DOC_TYPE_GUIDE,
    DOC_TYPE_INSPECTION_NOTICE,
    DOC_TYPE_INVOICE,
    DOC_TYPE_LEASE,
    DOC_TYPE_MAINTENANCE_LOG,
    DOC_TYPE_NOTICE,
    DOC_TYPE_POLICY,
    DOC_TYPE_RENEWAL_OFFER,
    DOC_TYPE_RENT_LEDGER,
    DOC_TYPE_REPORT,
    DOC_TYPE_WATER_BILL,
)
from app.enrichment import DocumentMetadataEnricher


def make_node(text="sample text", file_name="test.pdf", file_path="/docs/test.pdf"):
    node = types.SimpleNamespace()
    node.get_content = lambda **_: text
    node.metadata = {"file_name": file_name, "file_path": file_path}
    return node


class TestClassifyByRules:
    def test_faq_requires_both_keywords(self):
        assert DocumentMetadataEnricher._classify_by_rules("q: what?") is None
        assert DocumentMetadataEnricher._classify_by_rules("a: this") is None
        assert DocumentMetadataEnricher._classify_by_rules("q: what?\na: this") == DOC_TYPE_FAQ

    def test_inspection_notice(self):
        assert DocumentMetadataEnricher._classify_by_rules("routine inspection notice") == DOC_TYPE_INSPECTION_NOTICE

    def test_renewal_offer(self):
        assert DocumentMetadataEnricher._classify_by_rules("lease renewal offer") == DOC_TYPE_RENEWAL_OFFER

    def test_bond_lodgement(self):
        assert DocumentMetadataEnricher._classify_by_rules("bond lodgement confirmation") == DOC_TYPE_BOND_LODGEMENT

    def test_rent_ledger(self):
        assert DocumentMetadataEnricher._classify_by_rules("rent ledger for tenant") == DOC_TYPE_RENT_LEDGER

    def test_water_bill(self):
        assert DocumentMetadataEnricher._classify_by_rules("water usage kilolitres") == DOC_TYPE_WATER_BILL

    def test_maintenance_log(self):
        assert DocumentMetadataEnricher._classify_by_rules("maintenance request submitted") == DOC_TYPE_MAINTENANCE_LOG

    def test_notice(self):
        assert DocumentMetadataEnricher._classify_by_rules("notice to vacate") == DOC_TYPE_NOTICE

    def test_lease(self):
        assert DocumentMetadataEnricher._classify_by_rules("residential tenancy agreement") == DOC_TYPE_LEASE

    def test_invoice(self):
        assert DocumentMetadataEnricher._classify_by_rules("invoice amount due") == DOC_TYPE_INVOICE

    def test_report(self):
        assert DocumentMetadataEnricher._classify_by_rules("market report valuation") == DOC_TYPE_REPORT

    def test_policy(self):
        assert DocumentMetadataEnricher._classify_by_rules("agency policy terms") == DOC_TYPE_POLICY

    def test_guide(self):
        assert DocumentMetadataEnricher._classify_by_rules("step 1 guide suburb") == DOC_TYPE_GUIDE

    def test_unmatched_returns_none(self):
        assert DocumentMetadataEnricher._classify_by_rules("random unrelated content") is None

    def test_case_insensitive(self):
        assert DocumentMetadataEnricher._classify_by_rules("WATER BILL kilolitres") == DOC_TYPE_WATER_BILL

    def test_specific_type_wins_over_general(self):
        # inspection_notice is ordered before notice in CLASSIFICATION_RULES
        text = "inspection notice notice to vacate"
        assert DocumentMetadataEnricher._classify_by_rules(text) == DOC_TYPE_INSPECTION_NOTICE


class TestClassifyDocument:
    enricher = DocumentMetadataEnricher()

    def test_manifest_entry_takes_priority_over_rules(self):
        manifest_entry = {"document_type": "policy"}
        result = self.enricher._classify_document(
            "routine inspection notice", manifest_entry=manifest_entry
        )
        assert result == "policy"

    def test_rule_based_fallback_when_no_manifest(self):
        result = self.enricher._classify_document(
            "routine inspection notice", manifest_entry={}
        )
        assert result == DOC_TYPE_INSPECTION_NOTICE

    @patch("app.enrichment.classify_with_llama", return_value="invoice")
    def test_llm_fallback_when_no_rule_match(self, _):
        result = self.enricher._classify_document("random text", manifest_entry={})
        assert result == DOC_TYPE_INVOICE

    @patch("app.enrichment.classify_with_llama", return_value="not_a_real_type")
    def test_llm_invalid_type_falls_back_to_generic(self, _):
        result = self.enricher._classify_document("random text", manifest_entry={})
        assert result == DOC_TYPE_GENERIC

    @patch("app.enrichment.classify_with_llama", return_value="")
    def test_llm_empty_response_falls_back_to_generic(self, _):
        result = self.enricher._classify_document("random text", manifest_entry={})
        assert result == DOC_TYPE_GENERIC

    def test_empty_text_returns_generic(self):
        result = self.enricher._classify_document("", manifest_entry={})
        assert result == DOC_TYPE_GENERIC


class TestDocumentMetadataEnricherCall:

    def test_metadata_fields_stamped_on_node(self):
        mock_settings = MagicMock()
        mock_settings.AGENCY_ID = "AGN-001"

        # Use a filename not present in the real manifest so manifest_entry is always {}
        node = make_node(
            text="routine inspection notice",
            file_name="__test_not_in_manifest__.pdf",
            file_path="/docs/test.pdf",
        )
        with patch("app.enrichment.settings", mock_settings):
            result = DocumentMetadataEnricher()([node])

        meta = result[0].metadata
        assert meta["doc_type"] == DOC_TYPE_INSPECTION_NOTICE
        assert "doc_id" in meta and len(meta["doc_id"]) == 36  # UUID format
        assert meta["source"] == "/docs/test.pdf"
        assert meta["agency_id"] == ""     # manifest_entry is {} → default ""
        assert meta["property_id"] == ""

    def test_manifest_extra_fields_added(self):
        mock_settings = MagicMock()
        mock_settings.AGENCY_ID = "AGN-001"

        node = make_node(file_name="lease.pdf")
        manifest = {
            "lease.pdf": {
                "document_type": "lease",
                "agency_id": "AGN-001",
                "property_id": "P-42",
                "lease_start": "2024-01-01",
            }
        }
        with patch("app.enrichment.settings", mock_settings), \
             patch.object(DocumentMetadataEnricher, "_manifest", manifest):
            result = DocumentMetadataEnricher()([node])

        meta = result[0].metadata
        assert meta["doc_type"] == "lease"
        assert meta["agency_id"] == "AGN-001"
        assert meta["property_id"] == "P-42"
        assert meta["lease_start"] == "2024-01-01"

    def test_exception_during_enrichment_defaults_to_generic(self):
        mock_settings = MagicMock()
        mock_settings.AGENCY_ID = "AGN-001"

        node = make_node()
        enricher = DocumentMetadataEnricher()
        with patch("app.enrichment.settings", mock_settings), \
             patch.object(DocumentMetadataEnricher, "_manifest", {}), \
             patch.object(enricher, "_classify_document", side_effect=RuntimeError("boom")):
            result = enricher([node])

        assert result[0].metadata["doc_type"] == DOC_TYPE_GENERIC

    def test_multiple_nodes_all_enriched(self):
        mock_settings = MagicMock()
        mock_settings.AGENCY_ID = "AGN-001"

        nodes = [make_node(file_name=f"doc{i}.pdf") for i in range(3)]
        with patch("app.enrichment.settings", mock_settings), \
             patch.object(DocumentMetadataEnricher, "_manifest", {}):
            result = DocumentMetadataEnricher()(nodes)

        assert len(result) == 3
        assert all("doc_type" in n.metadata for n in result)
