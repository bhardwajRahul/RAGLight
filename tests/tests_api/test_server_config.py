import os
from unittest.mock import patch

import pytest

from raglight.api.server_config import ServerConfig
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.config.settings import Settings


def test_defaults():
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("RAGLIGHT_")}
    with patch.dict(os.environ, clean_env, clear=True):
        cfg = ServerConfig()
    assert cfg.llm_model == Settings.DEFAULT_LLM
    assert cfg.llm_provider == Settings.OLLAMA
    assert cfg.embeddings_model == Settings.DEFAULT_EMBEDDINGS_MODEL
    assert cfg.embeddings_provider == Settings.HUGGINGFACE
    assert cfg.collection == "default"
    assert cfg.k == Settings.DEFAULT_K
    assert cfg.chroma_host is None
    assert cfg.chroma_port is None


def test_env_override():
    env = {
        "RAGLIGHT_LLM_MODEL": "mistral",
        "RAGLIGHT_LLM_PROVIDER": "Mistral",
        "RAGLIGHT_LLM_API_BASE": "http://mistral:11434",
        "RAGLIGHT_EMBEDDINGS_MODEL": "nomic-embed-text",
        "RAGLIGHT_EMBEDDINGS_PROVIDER": "Ollama",
        "RAGLIGHT_EMBEDDINGS_API_BASE": "http://ollama:11434",
        "RAGLIGHT_PERSIST_DIR": "/tmp/mydb",
        "RAGLIGHT_COLLECTION": "myproject",
        "RAGLIGHT_K": "10",
        "RAGLIGHT_CHROMA_HOST": "chromadb",
        "RAGLIGHT_CHROMA_PORT": "8001",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = ServerConfig()

    assert cfg.llm_model == "mistral"
    assert cfg.llm_provider == "Mistral"
    assert cfg.llm_api_base == "http://mistral:11434"
    assert cfg.embeddings_model == "nomic-embed-text"
    assert cfg.embeddings_provider == "Ollama"
    assert cfg.embeddings_api_base == "http://ollama:11434"
    assert cfg.persist_dir == "/tmp/mydb"
    assert cfg.collection == "myproject"
    assert cfg.k == 10
    assert cfg.chroma_host == "chromadb"
    assert cfg.chroma_port == 8001


def test_to_rag_config_returns_correct_type():
    cfg = ServerConfig()
    rag_config = cfg.to_rag_config()
    assert isinstance(rag_config, RAGConfig)
    assert rag_config.llm == cfg.llm_model
    assert rag_config.provider == cfg.llm_provider
    assert rag_config.k == cfg.k
    assert rag_config.system_prompt == cfg.system_prompt


def test_to_vector_store_config_returns_correct_type():
    cfg = ServerConfig()
    vs_config = cfg.to_vector_store_config()
    assert isinstance(vs_config, VectorStoreConfig)
    assert vs_config.embedding_model == cfg.embeddings_model
    assert vs_config.provider == cfg.embeddings_provider
    assert vs_config.collection_name == cfg.collection


def test_to_vector_store_config_chroma_host():
    env = {
        "RAGLIGHT_CHROMA_HOST": "chromadb",
        "RAGLIGHT_CHROMA_PORT": "8000",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = ServerConfig()
    vs_config = cfg.to_vector_store_config()
    assert vs_config.host == "chromadb"
    assert vs_config.port == 8000
    assert vs_config.persist_directory is None


def test_to_vector_store_config_persistent():
    env = {"RAGLIGHT_PERSIST_DIR": "/tmp/testdb"}
    with patch.dict(os.environ, {k: "" for k in os.environ if k.startswith("RAGLIGHT_CHROMA_")}, clear=False):
        for k in list(os.environ.keys()):
            if k.startswith("RAGLIGHT_CHROMA_"):
                os.environ.pop(k, None)
        with patch.dict(os.environ, env, clear=False):
            cfg = ServerConfig()
    vs_config = cfg.to_vector_store_config()
    assert vs_config.persist_directory == "/tmp/testdb"
    assert vs_config.host is None
