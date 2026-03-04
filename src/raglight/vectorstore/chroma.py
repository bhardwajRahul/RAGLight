from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, cast
from typing_extensions import override

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_core.documents import Document

from ..document_processing.document_processor import DocumentProcessor
from .vector_store import VectorStore
from .bm25_index import BM25Index
from ..embeddings.embeddings_model import EmbeddingsModel


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """
    Adapter to make EmbeddingsModel compatible with ChromaDB's EmbeddingFunction interface.
    """

    def __init__(self, embeddings_model: EmbeddingsModel):
        self.embeddings_model = embeddings_model

    def __call__(self, input: Documents) -> Embeddings:
        if hasattr(self.embeddings_model, "embed_documents"):
            return self.embeddings_model.embed_documents(cast(List[str], input))
        else:
            raise TypeError(
                f"Object {type(self.embeddings_model)} does not implement 'embed_documents'."
            )


class ChromaVS(VectorStore):
    """
    Concrete implementation for ChromaDB using the official chromadb library.
    """

    def __init__(
        self,
        collection_name: str,
        embeddings_model: EmbeddingsModel,
        persist_directory: str = None,
        custom_processors: Optional[Dict[str, DocumentProcessor]] = None,
        host: str = None,
        port: int = None,
        search_type: str = "semantic",
        alpha: float = 0.5,
    ) -> None:
        super().__init__(persist_directory, embeddings_model, custom_processors)

        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.search_type = search_type
        self.alpha = alpha

        self.embedding_function = ChromaEmbeddingAdapter(self.embeddings_model)

        if host and port:
            self.client = chromadb.HttpClient(host=host, port=port, ssl=False)
        elif persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            raise ValueError(
                "Invalid configuration: provide host/port OR persist_directory."
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        self.collection_classes = self.client.get_or_create_collection(
            name=f"{collection_name}_classes",
            embedding_function=self.embedding_function,
        )

        self._bm25 = BM25Index()
        bm25_path = self._bm25_path()
        if bm25_path and bm25_path.exists():
            self._bm25.load(bm25_path)
        elif search_type in ("bm25", "hybrid"):
            self._rebuild_bm25_from_chroma()

    def _bm25_path(self) -> Optional[Path]:
        if not self.persist_directory:
            return None
        return Path(self.persist_directory) / f"bm25_{self.collection_name}.json"

    def _rebuild_bm25_from_chroma(self) -> None:
        result = self.collection.get()
        texts = result.get("documents") or []
        if texts:
            self._bm25.add_documents(texts)

    @override
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} document chunks to ChromaDB collection '{self.collection.name}'..."
        )

        self._add_docs_to_collection(self.collection, documents)

        texts = [doc.page_content for doc in documents]
        self._bm25.add_documents(texts)
        bm25_path = self._bm25_path()
        if bm25_path:
            self._bm25.save(bm25_path)

        logging.info("✅ Documents successfully added to the main collection.")

    @override
    def add_class_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logging.info(
            f"⏳ Adding {len(documents)} class documents to ChromaDB collection '{self.collection_classes.name}'..."
        )

        self._add_docs_to_collection(self.collection_classes, documents)

        logging.info("✅ Class documents successfully added to the class collection.")

    def _add_docs_to_collection(
        self, collection: Any, documents: List[Document]
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [
            doc.metadata if isinstance(doc.metadata, dict) else {} for doc in documents
        ]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def _bm25_search(self, question: str, k: int) -> List[Document]:
        results = self._bm25.search(question, k)
        docs = []
        for idx, _score in results:
            if idx < len(self._bm25.corpus):
                docs.append(Document(page_content=self._bm25.corpus[idx]))
        return docs

    def _rrf(
        self, ranked_lists: List[List[Document]], k_rrf: int = 60
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.page_content[:100]
                scores[key] = scores.get(key, 0) + 1 / (k_rrf + rank + 1)
                doc_map[key] = doc
        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def _hybrid_search(
        self, question: str, k: int, filter: Optional[Dict[str, Any]]
    ) -> List[Document]:
        fetch_k = k * 2
        semantic_docs = self._query_collection(
            self.collection, question, fetch_k, filter
        )
        bm25_docs = self._bm25_search(question, fetch_k)
        return self._rrf([semantic_docs, bm25_docs])[:k]

    @override
    def similarity_search(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        if self.search_type == "bm25":
            return self._bm25_search(question, k)
        elif self.search_type == "hybrid":
            return self._hybrid_search(question, k, filter)

        # semantic (default)
        target_collection = self.collection
        if collection_name and collection_name != self.collection.name:
            target_collection = self.client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
        return self._query_collection(target_collection, question, k, filter)

    @override
    def similarity_search_class(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        target_collection = self.collection_classes

        if collection_name:
            class_col_name = f"{collection_name}_classes"
            if class_col_name != self.collection_classes.name:
                target_collection = self.client.get_or_create_collection(
                    name=class_col_name, embedding_function=self.embedding_function
                )

        return self._query_collection(target_collection, question, k, filter)

    def _query_collection(
        self, collection: Any, question: str, k: int, filter: Optional[Dict[str, Any]]
    ) -> List[Document]:
        results = collection.query(query_texts=[question], n_results=k, where=filter)

        found_docs: List[Document] = []
        if results["documents"] and results["documents"][0]:
            docs_list = results["documents"][0]
            metas_list = (
                results["metadatas"][0]
                if results["metadatas"]
                else [{}] * len(docs_list)
            )
            for text, meta in zip(docs_list, metas_list):
                safe_meta = meta if isinstance(meta, dict) else {}
                found_docs.append(Document(page_content=text, metadata=safe_meta))

        return found_docs

    @override
    def get_available_collections(self) -> List[str]:
        """
        Retrieves the list of available collections in the ChromaDB.
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logging.error(f"Error listing collections: {e}")
            return []
