from unittest.mock import MagicMock, patch

import pytest

from app.enrichment import DocumentMetadataEnricher


def make_node(content: str, metadata: dict | None = None) -> MagicMock:
    node = MagicMock()
    node.get_content.return_value = content
    node.metadata = metadata or {}
    return node


class TestClassifyByRules:
    def test_faq_detected_by_qa_pattern(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Q: What is this? A: It is a test.") == "faq"

    def test_policy_detected_by_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("This is our privacy policy.") == "policy"

    def test_policy_detected_by_terms_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Terms and conditions apply.") == "policy"

    def test_guide_detected_by_step_pattern(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Step 1: Open the app.") == "guide"

    def test_guide_detected_by_guide_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("This guide explains how to proceed.") == "guide"

    def test_contract_detected_by_lease_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("This lease agreement is valid for 12 months.") == "contract"

    def test_contract_detected_by_tenant_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("The tenant must pay rent on time.") == "contract"

    def test_contract_detected_by_landlord_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("The landlord agrees to maintain the property.") == "contract"

    def test_contract_detected_by_bond_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("A bond of $2000 is required.") == "contract"

    def test_report_detected_by_valuation_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Property valuation as of January 2024.") == "report"

    def test_report_detected_by_market_report_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Q4 market report for Sydney suburbs.") == "report"

    def test_report_detected_by_inspection_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Inspection completed on 01/03/2024.") == "report"

    def test_notice_detected_by_vacate_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("You are required to vacate the premises.") == "notice"

    def test_notice_detected_by_notice_to_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Notice to tenant regarding unpaid rent.") == "notice"

    def test_notice_detected_by_breach_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("This is a breach notice issued by the agent.") == "notice"

    def test_invoice_detected_by_invoice_keyword(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Invoice #1023 for maintenance work.") == "invoice"

    def test_invoice_detected_by_amount_due(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Amount due: $450.00") == "invoice"

    def test_invoice_detected_by_total(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Total: $1200 payable within 14 days.") == "invoice"

    def test_returns_none_for_unknown_content(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Some random unrelated content.") is None

    def test_case_insensitive(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("POLICY document here.") == "policy"

    def test_contract_takes_priority_over_policy(self):
        """'agreement' and 'terms' both present — contract should win."""
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_by_rules("Lease agreement terms apply.") == "contract"


class TestClassifyDocument:
    def test_empty_text_returns_generic(self):
        enricher = DocumentMetadataEnricher()
        assert enricher._classify_document("") == "generic"

    def test_manifest_hit_returns_correct_type(self):
        enricher = DocumentMetadataEnricher()
        with patch.object(DocumentMetadataEnricher, "_manifest", {"suburb-guides.pdf": "guide"}):
            result = enricher._classify_document("lease bond tenant landlord", "suburb-guides.pdf")
            assert result == "guide"

    def test_manifest_takes_priority_over_rules(self):
        enricher = DocumentMetadataEnricher()
        with patch.object(DocumentMetadataEnricher, "_manifest", {"fee-schedule.pdf": "invoice"}):
            with patch("app.enrichment.classify_with_llama") as mock_llm:
                result = enricher._classify_document("lease bond tenant", "fee-schedule.pdf")
                mock_llm.assert_not_called()
                assert result == "invoice"

    def test_manifest_miss_falls_through_to_rules(self):
        enricher = DocumentMetadataEnricher()
        with patch.object(DocumentMetadataEnricher, "_manifest", {}):
            result = enricher._classify_document("Q: Hello A: World", "unknown.pdf")
            assert result == "faq"

    def test_empty_filename_skips_manifest(self):
        enricher = DocumentMetadataEnricher()
        with patch.object(DocumentMetadataEnricher, "_manifest", {"suburb-guides.pdf": "guide"}):
            result = enricher._classify_document("Q: Hello A: World", "")
            assert result == "faq"

    def test_rule_based_takes_priority_over_llm(self):
        enricher = DocumentMetadataEnricher()
        with patch("app.enrichment.classify_with_llama") as mock_llm:
            result = enricher._classify_document("Q: Hello A: World")
            mock_llm.assert_not_called()
            assert result == "faq"

    def test_llm_fallback_called_when_no_rule_match(self):
        enricher = DocumentMetadataEnricher()
        with patch("app.enrichment.classify_with_llama", return_value="guide") as mock_llm:
            result = enricher._classify_document("Some generic content here.")
            mock_llm.assert_called_once()
            assert result == "guide"

    def test_invalid_llm_result_falls_back_to_generic(self):
        enricher = DocumentMetadataEnricher()
        with patch("app.enrichment.classify_with_llama", return_value="unknown_type"):
            result = enricher._classify_document("Some content.")
            assert result == "generic"

    def test_none_llm_result_falls_back_to_generic(self):
        enricher = DocumentMetadataEnricher()
        with patch("app.enrichment.classify_with_llama", return_value=None):
            result = enricher._classify_document("Some content.")
            assert result == "generic"

    def test_llm_result_is_normalized(self):
        enricher = DocumentMetadataEnricher()
        with patch("app.enrichment.classify_with_llama", return_value="  POLICY  "):
            result = enricher._classify_document("Some content.")
            assert result == "policy"

    def test_llm_can_return_new_doc_types(self):
        enricher = DocumentMetadataEnricher()
        for doc_type in ("contract", "report", "notice", "invoice"):
            with patch("app.enrichment.classify_with_llama", return_value=doc_type):
                result = enricher._classify_document("Some content.")
                assert result == doc_type

    def test_text_truncated_to_max_chars_before_llm(self):
        enricher = DocumentMetadataEnricher()
        long_text = "x" * 5000
        with patch("app.enrichment.classify_with_llama", return_value="generic") as mock_llm:
            enricher._classify_document(long_text)
            called_with = mock_llm.call_args[0][0]
            assert len(called_with) == DocumentMetadataEnricher.MAX_CLASSIFICATION_CHARS


class TestEnricherCall:
    def test_doc_type_added_to_node_metadata(self):
        enricher = DocumentMetadataEnricher()
        node = make_node("This is our policy document.")
        enricher([node])
        assert node.metadata["doc_type"] == "policy"

    def test_source_set_from_file_path(self):
        enricher = DocumentMetadataEnricher()
        node = make_node("Step 1: do this.", metadata={"file_path": "/docs/guide.txt"})
        enricher([node])
        assert node.metadata["source"] == "/docs/guide.txt"

    def test_source_not_overwritten_if_already_set(self):
        enricher = DocumentMetadataEnricher()
        node = make_node("Step 1: do this.", metadata={"file_path": "/new.txt", "source": "/original.txt"})
        enricher([node])
        assert node.metadata["source"] == "/original.txt"

    def test_agency_id_added_to_node_metadata(self):
        enricher = DocumentMetadataEnricher()
        node = make_node("This is our policy document.")
        enricher([node])
        assert "agency_id" in node.metadata

    def test_agency_name_added_to_node_metadata(self):
        enricher = DocumentMetadataEnricher()
        node = make_node("This is our policy document.")
        enricher([node])
        assert "agency_name" in node.metadata

    def test_exception_falls_back_to_generic(self):
        enricher = DocumentMetadataEnricher()
        node = MagicMock()
        node.get_content.side_effect = RuntimeError("boom")
        node.metadata = {}
        enricher([node])
        assert node.metadata["doc_type"] == "generic"

    def test_multiple_nodes_all_tagged(self):
        enricher = DocumentMetadataEnricher()
        nodes = [
            make_node("Q: Hi A: Hello"),
            make_node("This is our terms and conditions."),
            make_node("The tenant must pay bond of $1500."),
            make_node("Notice to vacate by end of month."),
            make_node("Invoice #42 — Amount due: $300"),
            make_node("Inspection report completed."),
        ]
        enricher(nodes)
        assert nodes[0].metadata["doc_type"] == "faq"
        assert nodes[1].metadata["doc_type"] == "policy"
        assert nodes[2].metadata["doc_type"] == "contract"
        assert nodes[3].metadata["doc_type"] == "notice"
        assert nodes[4].metadata["doc_type"] == "invoice"
        assert nodes[5].metadata["doc_type"] == "report"
