from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_embedding_service():
    """Returns an EmbeddingService with a mocked HuggingFace model."""
    with patch("app.embeddings.embedding_service.HuggingFaceEmbedding") as MockModel:
        mock_model = MagicMock()
        MockModel.return_value = mock_model

        from app.embeddings.embedding_service import EmbeddingService
        service = EmbeddingService()
        service._model = mock_model
        yield service, mock_model


class TestEmbedText:
    def test_returns_embedding_for_valid_text(self, mock_embedding_service):
        service, mock_model = mock_embedding_service
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        result = service.embed_text("hello world")

        mock_model.get_text_embedding.assert_called_once_with("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_raises_for_empty_text(self, mock_embedding_service):
        service, _ = mock_embedding_service
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            service.embed_text("   ")

    def test_raises_for_blank_string(self, mock_embedding_service):
        service, _ = mock_embedding_service
        with pytest.raises(ValueError):
            service.embed_text("")


class TestEmbedBatch:
    def test_returns_embeddings_for_multiple_texts(self, mock_embedding_service):
        service, mock_model = mock_embedding_service
        mock_model.get_text_embedding_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        result = service.embed_batch(["text one", "text two"])

        mock_model.get_text_embedding_batch.assert_called_once_with(["text one", "text two"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_returns_empty_list_for_empty_input(self, mock_embedding_service):
        service, mock_model = mock_embedding_service

        result = service.embed_batch([])

        mock_model.get_text_embedding_batch.assert_not_called()
        assert result == []

    def test_passes_sequence_as_list(self, mock_embedding_service):
        service, mock_model = mock_embedding_service
        mock_model.get_text_embedding_batch.return_value = [[0.5]]

        service.embed_batch(("only one",))  # tuple input

        mock_model.get_text_embedding_batch.assert_called_once_with(["only one"])
