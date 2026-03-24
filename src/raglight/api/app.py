import logging
import os

# Disable ChromaDB telemetry before chromadb is imported
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Disable Langfuse OTLP auto-instrumentation when not configured.
# Langfuse v4 auto-initializes its OpenTelemetry exporter on import and tries
# to reach localhost:3000 by default even without explicit credentials.
_langfuse_configured = all([
    os.environ.get("LANGFUSE_PUBLIC_KEY"),
    os.environ.get("LANGFUSE_SECRET_KEY"),
    os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL"),
])
if not _langfuse_configured:
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"))

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..rag.simple_rag_api import RAGPipeline
from .router import create_router
from .server_config import ServerConfig

# Silence noisy third-party loggers (runs in the uvicorn subprocess, where CLI callback() never fires)
for _logger_name in [
    "telemetry",
    "langchain",
    "langchain_core",
    "langchain_core.tracing",
    "httpx",
    "urllib3",
    "requests",
    "chromadb",
    "chromadb.telemetry",
    "chromadb.telemetry.product.posthog",
]:
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL + 1)


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    if config is None:
        config = ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline = RAGPipeline(config.to_rag_config(), config.to_vector_store_config())
        app.state.pipeline = pipeline
        app.state.server_config = config
        yield

    app = FastAPI(
        title="RAGLight API",
        description="REST API for RAGLight RAG pipeline",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(create_router())
    return app
