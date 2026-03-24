from __future__ import annotations
import logging
import uuid
from typing import List, Dict, Optional, Any
from typing_extensions import override

from langchain_core.documents import Document

from ..document_processing.document_processor import DocumentProcessor
from .vector_store import VectorStore
from ..embeddings.embeddings_model import EmbeddingsModel


class QdrantVS(VectorStore):
    """
    Concrete implementation for Qdrant using the qdrant-client library.

    Supports local (on-disk) and remote (HTTP) modes.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
        persist_directory: str = None,
        custom_processors: Optional[Dict[str, DocumentProcessor]] = None,
        host: str = None,
        port: int = 6333,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required to use QdrantVS. "
                "Install it with: pip install raglight[qdrant]"
            )

        super().__init__(persist_directory, embeddings_model, custom_processors)

        self.collection_name = collection_name
        self._classes_collection_name = f"{collection_name}_classes"

        if host:
            self.client = QdrantClient(host=host, port=port)
        elif persist_directory:
            self.client = QdrantClient(path=persist_directory)
        else:
            raise ValueError(
                "Invalid configuration: provide host OR persist_directory."
            )

        # Probe vector dimension using a dummy embedding
        sample = embeddings_model.embed_query("probe")
        self._vector_size = len(sample)

        self._ensure_collection(self.collection_name)
        self._ensure_collection(self._classes_collection_name)

    def _ensure_collection(self, name: str) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def _add_to_collection(
        self, collection_name: str, documents: List[Document]
    ) -> None:
        from qdrant_client.models import PointStruct

        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings_model.embed_documents(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": text,
                    **(doc.metadata if isinstance(doc.metadata, dict) else {}),
                },
            )
            for text, vector, doc in zip(texts, vectors, documents)
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def _search_collection(
        self,
        collection_name: str,
        question: str,
        k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[Document]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_vector = self.embeddings_model.embed_query(question)

        qdrant_filter = None
        if filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filter.items()
                ]
            )

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
        ).points

        docs = []
        for hit in results:
            payload = hit.payload or {}
            page_content = payload.pop("page_content", "")
            docs.append(Document(page_content=page_content, metadata=payload))
        return docs

    @override
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        logging.info(
            f"⏳ Adding {len(documents)} document chunks to Qdrant collection '{self.collection_name}'..."
        )
        self._add_to_collection(self.collection_name, documents)
        logging.info("✅ Documents successfully added.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        logging.info(
            f"⏳ Adding {len(documents)} class documents to Qdrant collection '{self._classes_collection_name}'..."
        )
        self._add_to_collection(self._classes_collection_name, documents)
        logging.info("✅ Class documents successfully added.")

    @override
    def similarity_search(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        target = collection_name or self.collection_name
        return self._search_collection(target, question, k, filter)

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        if collection_name:
            target = f"{collection_name}_classes"
        else:
            target = self._classes_collection_name
        return self._search_collection(target, question, k, filter)

    @override
    def get_available_collections(self) -> List[str]:
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            logging.error(f"Error listing Qdrant collections: {e}")
            return []
