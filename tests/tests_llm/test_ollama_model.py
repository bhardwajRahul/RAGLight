import unittest
from unittest.mock import MagicMock, patch

from ollama import ChatResponse, Message
from raglight.llm.ollama_model import OllamaModel
from ..test_config import TestsConfig


class TestOllamaModel(unittest.TestCase):
    def setUp(self):
        mock_ollama_client = MagicMock()

        self.model = OllamaModel(
            model_name=TestsConfig.OLLAMA_MODEL,
            system_prompt_file=TestsConfig.TEST_SYSTEM_PROMPT,
            preload_model=False,
            options={"temperature": 0.3},
            headers={"x-some-header": "some-value"},
        )

        message: Message = Message(
            role="assistant",
            content="Machine learning (ML) is a subset of artificial intelligence",
        )
        chat_response: ChatResponse = ChatResponse(
            message=message, prompt_eval_count=200, eval_count=50
        )
        mock_ollama_client.chat = MagicMock(return_value=chat_response)
        self.model.model = mock_ollama_client

    def test_generate_response(self):
        question = "Define machine learning."
        response = self.model.generate({"question": question})
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertGreater(len(response), 0, "Response should not be empty.")
        self.assertEqual(
            response, "Machine learning (ML) is a subset of artificial intelligence"
        )
        self.model.model.chat.assert_called_once_with(
            model=TestsConfig.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "This is just a test prompt"},
                {
                    "role": "user",
                    "content": "Define machine learning.",
                },
            ],
            options={"temperature": 0.3},
        )


class TestOllamaModelStreaming(unittest.TestCase):
    def setUp(self):
        self.model = OllamaModel(
            model_name=TestsConfig.OLLAMA_MODEL,
            system_prompt="You are helpful.",
            preload_model=False,
        )
        chunk1 = MagicMock()
        chunk1.message.content = "Hello"
        chunk2 = MagicMock()
        chunk2.message.content = " world"
        self.mock_client = MagicMock()
        self.mock_client.chat.return_value = iter([chunk1, chunk2])
        self.model.model = self.mock_client

    def test_generate_streaming_yields_chunks(self):
        result = list(self.model.generate_streaming({"question": "Say hello."}))
        self.assertEqual(result, ["Hello", " world"])

    def test_generate_streaming_builds_correct_messages(self):
        list(self.model.generate_streaming({"question": "Say hello."}))
        call_kwargs = self.mock_client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        # system prompt first
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")
        # user message with plain text content, NOT a JSON blob
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Say hello.")

    def test_generate_streaming_includes_history(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        list(
            self.model.generate_streaming(
                {"question": "Follow-up.", "history": history}
            )
        )
        call_kwargs = self.mock_client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        # system + 2 history + user question = 4
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "Previous question")

    def test_generate_streaming_passes_stream_true(self):
        list(self.model.generate_streaming({"question": "Say hello."}))
        call_kwargs = self.mock_client.chat.call_args.kwargs
        self.assertTrue(call_kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
