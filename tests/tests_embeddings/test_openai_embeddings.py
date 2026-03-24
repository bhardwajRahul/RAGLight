import unittest
from unittest.mock import MagicMock, patch

from raglight.embeddings.openai_embeddings import OpenAIEmbeddingsModel
from ..test_config import TestsConfig


class TestOpenAIEmbeddings(unittest.TestCase):

    @patch("raglight.embeddings.openai_embeddings.OpenAIEmbeddings")
    def test_model_load(self, MockEmbeddings: MagicMock):
        MockEmbeddings.return_value = MagicMock()
        model = OpenAIEmbeddingsModel(TestsConfig.OPENAI_EMBEDDING_MODEL)
        self.assertIsNotNone(model.model)
        MockEmbeddings.assert_called_once()

    @patch("raglight.embeddings.openai_embeddings.OpenAIEmbeddings")
    def test_embed_documents(self, MockEmbeddings: MagicMock):
        mock_instance = MockEmbeddings.return_value
        mock_instance.embed_documents.return_value = [[0.1, 0.1], [0.2, 0.2]]

        model = OpenAIEmbeddingsModel(TestsConfig.OPENAI_EMBEDDING_MODEL)
        result = model.embed_documents(["text1", "text2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.1])
        mock_instance.embed_documents.assert_called_with(["text1", "text2"])

    @patch("raglight.embeddings.openai_embeddings.OpenAIEmbeddings")
    def test_embed_query(self, MockEmbeddings: MagicMock):
        mock_instance = MockEmbeddings.return_value
        mock_instance.embed_query.return_value = [0.5, 0.5]

        model = OpenAIEmbeddingsModel(TestsConfig.OPENAI_EMBEDDING_MODEL)
        result = model.embed_query("query")

        self.assertEqual(result, [0.5, 0.5])
        mock_instance.embed_query.assert_called_with("query")


if __name__ == "__main__":
    unittest.main()
