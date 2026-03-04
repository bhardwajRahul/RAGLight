from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..rag.simple_rag_api import RAGPipeline
from .router import create_router
from .server_config import ServerConfig


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
