import json
from unittest.mock import MagicMock, patch

import pytest


class TestGenerateWithLLM:

    def _mock_llm_response(self, content: str) -> MagicMock:
        response = MagicMock()
        response.content = content
        return response

    @patch("app.suggestions.suggestion_service.llm")
    def test_valid_json_array_returned(self, mock_llm):
        questions = ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]
        mock_llm.invoke.return_value = self._mock_llm_response(json.dumps(questions))

        from app.suggestions.suggestion_service import _generate_with_llm
        result = _generate_with_llm("some document content")

        assert result == questions

    @patch("app.suggestions.suggestion_service.llm")
    def test_invalid_json_raises_runtime_error(self, mock_llm):
        mock_llm.invoke.return_value = self._mock_llm_response("not json at all")

        from app.suggestions.suggestion_service import _generate_with_llm
        with pytest.raises(RuntimeError, match="Failed to parse LLM suggestions response"):
            _generate_with_llm("some content")

    @patch("app.suggestions.suggestion_service.llm")
    def test_non_array_json_raises_runtime_error(self, mock_llm):
        mock_llm.invoke.return_value = self._mock_llm_response('{"key": "value"}')

        from app.suggestions.suggestion_service import _generate_with_llm
        with pytest.raises(RuntimeError):
            _generate_with_llm("some content")


class TestSavePayload:

    @patch("app.suggestions.suggestion_service._OUTPUT_DIR")
    def test_creates_file_with_correct_name(self, mock_dir, tmp_path):
        from pathlib import Path
        mock_dir.__truediv__ = lambda self, other: tmp_path / other
        mock_dir.mkdir = MagicMock()

        from app.suggestions.suggestion_service import _save_payload

        payload = {"docId": "doc-123", "suggestions": ["Q1?"]}
        _save_payload("lease.pdf", payload)

        out_file = tmp_path / "lease.pdf_suggestions.json"
        assert out_file.exists()
        assert json.loads(out_file.read_text()) == payload


class TestGenerateSuggestionsForDocument:

    @patch("app.suggestions.suggestion_service._fetch_document_data",
           return_value=([], "", "", ""))
    def test_no_chunks_raises_value_error(self, _):
        from app.suggestions.suggestion_service import generate_suggestions_for_document
        with pytest.raises(ValueError, match="No chunks found"):
            generate_suggestions_for_document("missing.pdf")

    @patch("app.suggestions.suggestion_service._save_payload")
    @patch("app.suggestions.suggestion_service._generate_with_llm",
           return_value=["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"])
    @patch("app.suggestions.suggestion_service._fetch_document_data",
           return_value=(["chunk1", "chunk2"], "doc-123", "AGN-001", "PROP-42"))
    def test_success_returns_correct_payload(self, _fetch, _llm, _save):
        mock_settings = MagicMock()
        mock_settings.LLAMA_MODEL_NAME = "llama3.2:3b"

        with patch("app.suggestions.suggestion_service.settings", mock_settings):
            from app.suggestions.suggestion_service import generate_suggestions_for_document
            result = generate_suggestions_for_document("lease.pdf")

        assert result["docId"] == "doc-123"
        assert result["agencyId"] == "AGN-001"
        assert result["fileName"] == "lease.pdf"
        assert result["propertyId"] == "PROP-42"
        assert result["suggestions"] == ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]
        assert result["modelUsed"] == "llama3.2:3b"
        _save.assert_called_once()

    @patch("app.suggestions.suggestion_service._fetch_document_data",
           return_value=(["chunk1"], "doc-123", "AGN-001", "PROP-42"))
    @patch("app.suggestions.suggestion_service._generate_with_llm",
           side_effect=RuntimeError("LLM failed"))
    def test_llm_failure_propagates(self, _llm, _fetch):
        from app.suggestions.suggestion_service import generate_suggestions_for_document
        with pytest.raises(RuntimeError, match="LLM failed"):
            generate_suggestions_for_document("lease.pdf")
