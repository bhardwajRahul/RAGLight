import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from raglight.rag.rag import RAG


def _make_rag(llm=None, reformulation=False):
    embedding_model = MagicMock()
    embedding_model.get_model.return_value = MagicMock()

    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [
        Document(page_content="RAGLight is a RAG library.", metadata={"source": "test"})
    ]

    if llm is None:
        llm = MagicMock()
        llm.generate.return_value = "RAGLight is a Python library."

    return RAG(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        k=2,
        reformulation=reformulation,
    )


class TestRAGGenerate(unittest.TestCase):
    def setUp(self):
        self.rag = _make_rag()

    def test_generate_returns_string(self):
        response = self.rag.generate("What is RAGLight?")
        self.assertIsInstance(response, str)
        self.assertEqual(response, "RAGLight is a Python library.")

    def test_generate_calls_similarity_search(self):
        self.rag.generate("What is RAGLight?")
        self.rag.vector_store.similarity_search.assert_called_once()

    def test_generate_calls_llm(self):
        self.rag.generate("What is RAGLight?")
        self.rag.llm.generate.assert_called_once()

    def test_generate_updates_history(self):
        question = "What is RAGLight?"
        response = self.rag.generate(question)
        history = self.rag.state["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": question})
        self.assertEqual(history[1], {"role": "assistant", "content": response})

    def test_generate_accumulates_history_across_turns(self):
        self.rag.generate("First question")
        self.rag.generate("Second question")
        self.assertEqual(len(self.rag.state["history"]), 4)

    def test_max_history_caps_history(self):
        self.rag.max_history = 2
        self.rag.generate("First question")
        self.rag.generate("Second question")
        self.rag.generate("Third question")
        self.assertLessEqual(len(self.rag.state["history"]), 4)


class TestRAGGenerateStreaming(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.llm.generate.return_value = "unused"
        self.llm.generate_streaming.return_value = iter(["RAGLight", " is", " a library."])
        self.rag = _make_rag(llm=self.llm)

    def test_generate_streaming_yields_chunks(self):
        chunks = list(self.rag.generate_streaming("What is RAGLight?"))
        self.assertEqual(chunks, ["RAGLight", " is", " a library."])

    def test_generate_streaming_calls_llm_streaming(self):
        list(self.rag.generate_streaming("What is RAGLight?"))
        self.llm.generate_streaming.assert_called_once()

    def test_generate_streaming_calls_similarity_search(self):
        list(self.rag.generate_streaming("What is RAGLight?"))
        self.rag.vector_store.similarity_search.assert_called_once()

    def test_generate_streaming_updates_history(self):
        question = "What is RAGLight?"
        chunks = list(self.rag.generate_streaming(question))
        full_answer = "".join(chunks)
        history = self.rag.state["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": question})
        self.assertEqual(history[1], {"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    unittest.main()
