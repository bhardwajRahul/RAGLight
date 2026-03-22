import asyncio
import logging
import shutil
import threading
from typing import List

from ..config.vector_store_config import VectorStoreConfig
from .agentic_rag import AgenticRAG
from ..config.agentic_rag_config import AgenticRAGConfig
from ..vectorstore.vector_store import VectorStore
from ..scrapper.github_scrapper import GithubScrapper
from ..models.data_source_model import DataSource, FolderSource, GitHubSource


class AgenticRAGPipeline:
    def __init__(
        self,
        config: AgenticRAGConfig,
        vector_store_config: VectorStoreConfig,
    ) -> None:
        """
        Initializes the AgenticRAGPipeline.

        A dedicated event loop running in a background daemon thread is created
        at construction time. All async calls are submitted to this loop via
        run_coroutine_threadsafe, which guarantees that LangGraph's internal
        async state (tasks, futures) is always executed in the same loop and
        never encounters a closed-loop error between calls.
        """
        self.config = config
        self.knowledge_base: List[DataSource] = config.knowledge_base
        self.ignore_folders = config.ignore_folders

        self.agenticRag = AgenticRAG(config, vector_store_config)
        self.github_scrapper: GithubScrapper = GithubScrapper()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def build(self) -> None:
        """
        Builds the RAG pipeline by ingesting data from the knowledge base.
        """
        repositories: List[GitHubSource] = []
        if not self.knowledge_base:
            return
        for source in self.knowledge_base:
            if isinstance(source, FolderSource):
                self.get_vector_store().ingest(
                    data_path=source.path, ignore_folders=self.ignore_folders
                )
            if isinstance(source, GitHubSource):
                repositories.append(source)
        if len(repositories) > 0:
            self.ingest_github_repositories(repositories)

    def ingest_github_repositories(self, repositories: List[GitHubSource]) -> None:
        """
        Clones and processes GitHub repositories for the pipeline.

        Args:
            repositories (List[GitHubSource]): A list of GitHub repository sources to clone and ingest.
        """
        self.github_scrapper.set_repositories(repositories)
        repos_path: str = self.github_scrapper.clone_all()
        self.get_vector_store().ingest(
            data_path=repos_path, ignore_folders=self.ignore_folders
        )
        shutil.rmtree(repos_path)
        logging.info("✅ GitHub repositories cleaned successfully!")

    def get_vector_store(self) -> VectorStore:
        return self.agenticRag.vector_store

    def clear_history(self) -> None:
        """Resets the agent's conversation history."""
        self.agenticRag.clear_history()

    def generate(self, question: str, stream: bool = False) -> str:
        """
        Synchronous wrapper for the agent's asynchronous generation.

        Submits the coroutine to the pipeline's persistent event loop running
        in a background thread. Works correctly in all calling contexts:
        plain scripts, FastAPI/uvicorn (run_in_threadpool), and Jupyter notebooks.
        """
        future = asyncio.run_coroutine_threadsafe(
            self.agenticRag.generate(question, stream=stream),
            self._loop,
        )
        return future.result()

    def close(self) -> None:
        """Stops the background event loop and joins its thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def __enter__(self) -> "AgenticRAGPipeline":
        return self

    def __exit__(self, *args) -> None:
        self.close()
