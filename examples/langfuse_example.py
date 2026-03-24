"""
Langfuse observability example for RAGLight.

Requirements:
    pip install "raglight[langfuse]"   # installs langfuse==4.0.0

Prerequisites:
    - A running Ollama instance (default: http://localhost:11434)
    - A running Langfuse instance, e.g. via Docker:
        docker run -p 3000:3000 langfuse/langfuse
      Or use Langfuse Cloud (https://cloud.langfuse.com) and update LANGFUSE_HOST below.

Usage:
    1. Set your Langfuse credentials in the LangfuseConfig below.
    2. Point DATA_PATH to a local folder containing documents to index.
    3. Run:  python examples/langfuse_example.py
    4. Open your Langfuse dashboard and watch the traces appear.
"""

import os

from raglight.config.langfuse_config import LangfuseConfig
from raglight.config.rag_config import RAGConfig
from raglight.config.settings import Settings
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.models.data_source_model import FolderSource
from raglight.rag.simple_rag_api import RAGPipeline

Settings.setup_logging()

# ---------------------------------------------------------------------------
# 1. Langfuse credentials
#    Replace with your actual keys (or load them from environment variables).
# ---------------------------------------------------------------------------
langfuse_config = LangfuseConfig(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-..."),
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    # session_id="my-fixed-session-id",  # optional: pin a custom trace ID
)

# ---------------------------------------------------------------------------
# 2. Vector store configuration
# ---------------------------------------------------------------------------
vector_store_config = VectorStoreConfig(
    embedding_model=Settings.DEFAULT_EMBEDDINGS_MODEL,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory="./langfuse_example_db",
    collection_name="langfuse_example",
)

# ---------------------------------------------------------------------------
# 3. RAG configuration — plug in langfuse_config here
# ---------------------------------------------------------------------------
DATA_PATH = os.environ.get("DATA_PATH", "./examples")

config = RAGConfig(
    llm=Settings.DEFAULT_LLM,
    provider=Settings.OLLAMA,
    k=3,
    knowledge_base=[FolderSource(path=DATA_PATH)],
    langfuse_config=langfuse_config,
)

# ---------------------------------------------------------------------------
# 4. Build the pipeline and run a few queries
# ---------------------------------------------------------------------------
pipeline = RAGPipeline(config, vector_store_config)
pipeline.build()

questions = [
    "What is RAGLight and what problem does it solve?",
    "How do I configure a custom LLM provider in RAGLight?",
]

for question in questions:
    print(f"\nQ: {question}")
    response = pipeline.generate(question)
    print(f"A: {response}")

print(
    "\nAll traces have been sent to Langfuse. " "Open your dashboard to inspect them."
)
