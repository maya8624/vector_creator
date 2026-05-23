import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.suggestions.suggestion_service import (
    _generate_with_llm,
    _fetch_document_data,
    _save_payload,
    generate_suggestions_for_document,
)

SAMPLE_SUGGESTIONS = [
    "What is the weekly rent?",
    "Is the property pet-friendly?",
    "What is the lease term?",
    "Is parking included?",
    "When is the property available?",
]


def make_mock_session(rows: list[tuple]) -> MagicMock:
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = rows
    return session


def patch_db(rows: list[tuple]):
    mock_db = MagicMock()
    mock_db.create_session.return_value.__enter__ = MagicMock(return_value=make_mock_session(rows))
    mock_db.create_session.return_value.__exit__ = MagicMock(return_value=False)
    return patch("app.suggestions.suggestion_service.PostgresService", return_value=mock_db)


def patch_metadata(doc_id: str = "", user_id: str = ""):
    return patch(
        "app.suggestions.suggestion_service._fetch_document_data",
        return_value=(["chunk"], doc_id, user_id),
    )


def patch_llm(response_text: str):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = response_text
    return patch("app.suggestions.suggestion_service.llm", mock_llm)



class TestGenerateWithLLM:
    def test_returns_parsed_suggestions(self):
        with patch_llm(json.dumps(SAMPLE_SUGGESTIONS)):
            result = _generate_with_llm("Some document content.")
        assert result == SAMPLE_SUGGESTIONS

    def test_strips_whitespace_before_parsing(self):
        with patch_llm(f"  {json.dumps(SAMPLE_SUGGESTIONS)}  "):
            result = _generate_with_llm("content")
        assert result == SAMPLE_SUGGESTIONS

    def test_raises_on_invalid_json(self):
        with patch_llm("not valid json at all"):
            with pytest.raises(RuntimeError, match="Failed to parse LLM suggestions response"):
                _generate_with_llm("content")

    def test_raises_when_response_is_object_not_array(self):
        with patch_llm(json.dumps({"question": "Why?"})):
            with pytest.raises(RuntimeError, match="Failed to parse LLM suggestions response"):
                _generate_with_llm("content")

    def test_raises_when_response_is_plain_string(self):
        with patch_llm('"just a string"'):
            with pytest.raises(RuntimeError):
                _generate_with_llm("content")


class TestSavePayload:
    def test_saves_json_file_with_correct_content(self, tmp_path: Path):
        payload = {"docId": "abc", "suggestions": SAMPLE_SUGGESTIONS}
        with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
            _save_payload("abc", payload)
        saved = json.loads((tmp_path / "abc.json").read_text(encoding="utf-8"))
        assert saved == payload

    def test_filename_matches_doc_id(self, tmp_path: Path):
        with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
            _save_payload("my-doc-id", {"docId": "my-doc-id"})
        assert (tmp_path / "my-doc-id.json").exists()

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "suggestions"
        with patch("app.suggestions.suggestion_service._OUTPUT_DIR", nested):
            _save_payload("doc", {"docId": "doc"})
        assert nested.is_dir()


class TestFetchDocumentData:
    def test_returns_chunks_doc_id_and_user_id(self):
        rows = [("chunk one", "doc-abc", "user-xyz"), ("chunk two", "doc-abc", "user-xyz")]
        with patch_db(rows):
            chunks, doc_id, user_id = _fetch_document_data("lease-template.pdf")
        assert chunks == ["chunk one", "chunk two"]
        assert doc_id == "doc-abc"
        assert user_id == "user-xyz"

    def test_returns_empty_when_no_rows(self):
        with patch_db([]):
            chunks, doc_id, user_id = _fetch_document_data("missing.pdf")
        assert chunks == []
        assert doc_id == ""
        assert user_id == ""

    def test_filters_empty_text_chunks(self):
        rows = [("valid", "doc-abc", "user-xyz"), ("", "doc-abc", "user-xyz")]
        with patch_db(rows):
            chunks, _, _ = _fetch_document_data("lease-template.pdf")
        assert chunks == ["valid"]


class TestGenerateSuggestionsForDocument:
    def test_returns_full_payload(self, tmp_path: Path):
        with patch_metadata("stored-doc-id", "stored-user-id"), patch_llm(json.dumps(SAMPLE_SUGGESTIONS)):
            with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
                result = generate_suggestions_for_document(file_name="lease-template.pdf")
        assert result["docId"] == "stored-doc-id"
        assert result["userId"] == "stored-user-id"
        assert result["suggestions"] == SAMPLE_SUGGESTIONS
        assert "modelUsed" in result

    def test_uses_doc_id_and_user_id_from_metadata(self, tmp_path: Path):
        with patch_metadata("stored-doc-id", "stored-user-id"), patch_llm(json.dumps(SAMPLE_SUGGESTIONS)):
            with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
                result = generate_suggestions_for_document(file_name="lease-template.pdf")
        assert result["docId"] == "stored-doc-id"
        assert result["userId"] == "stored-user-id"

    def test_saved_filename_matches_doc_id(self, tmp_path: Path):
        with patch_metadata("stored-doc-id", ""), patch_llm(json.dumps(SAMPLE_SUGGESTIONS)):
            with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
                generate_suggestions_for_document(file_name="lease-template.pdf")
        assert (tmp_path / "stored-doc-id.json").exists()

    def test_raises_when_no_chunks_found(self):
        with patch("app.suggestions.suggestion_service._fetch_document_data", return_value=([], "", "")):
            with pytest.raises(ValueError, match="No chunks found"):
                generate_suggestions_for_document(file_name="ghost.pdf")

    def test_payload_is_saved_to_disk(self, tmp_path: Path):
        with patch_metadata("saved-id", ""), patch_llm(json.dumps(SAMPLE_SUGGESTIONS)):
            with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
                generate_suggestions_for_document(file_name="lease-template.pdf")
        assert (tmp_path / "saved-id.json").exists()

    def test_doc_id_is_empty_when_not_in_metadata(self, tmp_path: Path):
        with patch_metadata(), patch_llm(json.dumps(SAMPLE_SUGGESTIONS)):
            with patch("app.suggestions.suggestion_service._OUTPUT_DIR", tmp_path):
                result = generate_suggestions_for_document(file_name="lease-template.pdf")
        assert result["docId"] == ""
