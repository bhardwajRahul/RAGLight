import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from raglight.api.router import create_router


def make_app(pipeline_mock: MagicMock) -> FastAPI:
    app = FastAPI()
    app.include_router(create_router())
    app.state.pipeline = pipeline_mock
    return app


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.pipeline = MagicMock()
        self.vector_store = MagicMock()
        self.pipeline.get_vector_store.return_value = self.vector_store
        self.client = TestClient(make_app(self.pipeline), raise_server_exceptions=True)

    # ── /health ──────────────────────────────────────────────────────────────

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    # ── /generate ─────────────────────────────────────────────────────────────

    def test_generate(self):
        self.pipeline.generate.return_value = "Paris is the capital of France."
        response = self.client.post("/generate", json={"question": "What is the capital of France?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "Paris is the capital of France."})
        self.pipeline.generate.assert_called_once_with("What is the capital of France?")

    def test_generate_error(self):
        self.pipeline.generate.side_effect = RuntimeError("LLM unavailable")
        response = self.client.post("/generate", json={"question": "test"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("LLM unavailable", response.json()["detail"])

    # ── /ingest ───────────────────────────────────────────────────────────────

    def test_ingest_local(self):
        self.vector_store.ingest.return_value = None
        response = self.client.post("/ingest", json={"data_path": "/some/local/path"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("success", response.json()["message"].lower())
        self.vector_store.ingest.assert_called_once_with(data_path="/some/local/path")

    def test_ingest_missing_params(self):
        response = self.client.post("/ingest", json={})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertTrue(any(k in detail for k in ("data_path", "file_paths", "github_url")))

    def test_ingest_github(self):
        self.vector_store.ingest.return_value = None
        with patch("raglight.api.router.GithubScrapper") as MockScrapper:
            instance = MagicMock()
            instance.clone_all.return_value = "/tmp/repos"
            MockScrapper.return_value = instance
            response = self.client.post(
                "/ingest",
                json={"github_url": "https://github.com/example/repo", "github_branch": "main"},
            )
        self.assertEqual(response.status_code, 200)
        instance.set_repositories.assert_called_once()
        instance.clone_all.assert_called_once()
        self.vector_store.ingest.assert_called_once_with(data_path="/tmp/repos")

    def test_ingest_file_paths(self):
        self.vector_store.add_documents.return_value = None
        self.vector_store.add_class_documents.return_value = None
        self.vector_store._flatten_metadata.side_effect = lambda docs: docs
        self.vector_store._process_file = lambda fp, factory, flatten: (["chunk"], [])
        with patch("os.path.isfile", return_value=True):
            response = self.client.post(
                "/ingest",
                json={"file_paths": ["/some/file.pdf", "/some/other.txt"]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("success", response.json()["message"].lower())

    def test_ingest_file_paths_missing_file(self):
        with patch("os.path.isfile", return_value=False):
            response = self.client.post("/ingest", json={"file_paths": ["/nonexistent/file.pdf"]})
        self.assertEqual(response.status_code, 500)
        self.assertIn("not found", response.json()["detail"].lower())

    # ── /ingest/upload ────────────────────────────────────────────────────────

    def test_ingest_upload(self):
        self.vector_store.add_documents.return_value = None
        self.vector_store.add_class_documents.return_value = None
        self.vector_store._flatten_metadata.side_effect = lambda docs: docs
        self.vector_store._process_file = lambda fp, factory, flatten: (["chunk"], [])
        response = self.client.post(
            "/ingest/upload",
            files=[
                ("files", ("doc1.txt", b"hello world", "text/plain")),
                ("files", ("doc2.txt", b"another doc", "text/plain")),
            ],
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("2 file(s)", response.json()["message"])

    def test_ingest_upload_no_files(self):
        response = self.client.post("/ingest/upload", files=[])
        self.assertIn(response.status_code, (400, 422))

    # ── /collections ──────────────────────────────────────────────────────────

    def test_collections(self):
        self.vector_store.get_available_collections.return_value = ["default", "project_x"]
        response = self.client.get("/collections")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"collections": ["default", "project_x"]})


if __name__ == "__main__":
    unittest.main()
