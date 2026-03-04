from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from raglight.api.router import create_router


def make_app(pipeline_mock: MagicMock) -> FastAPI:
    """Create a minimal FastAPI app with a mocked pipeline in state."""
    app = FastAPI()
    app.include_router(create_router())
    app.state.pipeline = pipeline_mock
    return app


@pytest.fixture
def pipeline():
    mock = MagicMock()
    vector_store_mock = MagicMock()
    mock.get_vector_store.return_value = vector_store_mock
    return mock


@pytest.fixture
def client(pipeline):
    app = make_app(pipeline)
    return TestClient(app, raise_server_exceptions=True)


# ── /health ──────────────────────────────────────────────────────────────────

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── /generate ─────────────────────────────────────────────────────────────────

def test_generate(client, pipeline):
    pipeline.generate.return_value = "Paris is the capital of France."
    response = client.post("/generate", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Paris is the capital of France."}
    pipeline.generate.assert_called_once_with("What is the capital of France?")


def test_generate_error(client, pipeline):
    pipeline.generate.side_effect = RuntimeError("LLM unavailable")
    response = client.post("/generate", json={"question": "test"})
    assert response.status_code == 500
    assert "LLM unavailable" in response.json()["detail"]


# ── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_local(client, pipeline):
    vector_store = pipeline.get_vector_store.return_value
    vector_store.ingest.return_value = None

    response = client.post("/ingest", json={"data_path": "/some/local/path"})
    assert response.status_code == 200
    assert "success" in response.json()["message"].lower()
    vector_store.ingest.assert_called_once_with(data_path="/some/local/path")


def test_ingest_missing_params(client):
    response = client.post("/ingest", json={})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert any(k in detail for k in ("data_path", "file_paths", "github_url"))


def test_ingest_github(client, pipeline):
    vector_store = pipeline.get_vector_store.return_value
    vector_store.ingest.return_value = None

    with patch("raglight.api.router.GithubScrapper") as MockScrapper:
        scrapper_instance = MagicMock()
        scrapper_instance.clone_all.return_value = "/tmp/repos"
        MockScrapper.return_value = scrapper_instance

        response = client.post(
            "/ingest",
            json={"github_url": "https://github.com/example/repo", "github_branch": "main"},
        )

    assert response.status_code == 200
    scrapper_instance.set_repositories.assert_called_once()
    scrapper_instance.clone_all.assert_called_once()
    vector_store.ingest.assert_called_once_with(data_path="/tmp/repos")


# ── /collections ──────────────────────────────────────────────────────────────

def test_ingest_file_paths(client, pipeline):
    vector_store = pipeline.get_vector_store.return_value
    vector_store.add_documents.return_value = None
    vector_store.add_class_documents.return_value = None
    vector_store._flatten_metadata.side_effect = lambda docs: docs
    vector_store._process_file = lambda fp, factory, flatten: (["chunk"], [])

    with patch("os.path.isfile", return_value=True):
        response = client.post(
            "/ingest",
            json={"file_paths": ["/some/file.pdf", "/some/other.txt"]},
        )

    assert response.status_code == 200
    assert "success" in response.json()["message"].lower()


def test_ingest_upload(client, pipeline):
    vector_store = pipeline.get_vector_store.return_value
    vector_store.add_documents.return_value = None
    vector_store.add_class_documents.return_value = None
    vector_store._flatten_metadata.side_effect = lambda docs: docs
    vector_store._process_file = lambda fp, factory, flatten: (["chunk"], [])

    response = client.post(
        "/ingest/upload",
        files=[
            ("files", ("doc1.txt", b"hello world", "text/plain")),
            ("files", ("doc2.txt", b"another doc", "text/plain")),
        ],
    )
    assert response.status_code == 200
    assert "2 file(s)" in response.json()["message"]


def test_ingest_upload_no_files(client):
    response = client.post("/ingest/upload", files=[])
    assert response.status_code in (400, 422)


def test_ingest_file_paths_missing_file(client, pipeline):
    with patch("os.path.isfile", return_value=False):
        response = client.post(
            "/ingest",
            json={"file_paths": ["/nonexistent/file.pdf"]},
        )
    assert response.status_code == 500
    assert "not found" in response.json()["detail"].lower()


def test_collections(client, pipeline):
    vector_store = pipeline.get_vector_store.return_value
    vector_store.get_available_collections.return_value = ["default", "project_x"]

    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json() == {"collections": ["default", "project_x"]}
