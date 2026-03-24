"""
Hybrid Search Example
=====================
Demonstrates the three search modes available in RAGLight:
  - "semantic" : vector similarity only (default)
  - "bm25"     : keyword-based BM25 search only
  - "hybrid"   : BM25 + semantic combined via Reciprocal Rank Fusion (RRF)

Requirements:
  - Ollama running locally with llama3 (or any model you prefer)
  - rank_bm25 installed (pip install raglight includes it)
"""

import uuid
from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

persist_directory = "./hybridDb"
model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL
model_name = "llama3.1:8b"
collection_name = str(uuid.uuid4())
data_path = "./src/raglight"  # folder to ingest — adjust to your own documents

# ── 1. Semantic search (default behaviour) ──────────────────────────────────
print("\n=== Semantic search ===")
rag_semantic = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings)
    .with_vector_store(
        Settings.CHROMA,
        persist_directory=persist_directory,
        collection_name=collection_name,
        search_type=Settings.SEARCH_SEMANTIC,  # default — can be omitted
    )
    .with_llm(
        Settings.OLLAMA,
        model_name=model_name,
        system_prompt=Settings.DEFAULT_SYSTEM_PROMPT,
    )
    .build_rag(k=5)
)
rag_semantic.vector_store.ingest(data_path=data_path)
response = rag_semantic.generate("How do I create a RAG pipeline with RAGLight?")
print(response)

# ── 2. BM25-only search ──────────────────────────────────────────────────────
print("\n=== BM25 search ===")
rag_bm25 = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings)
    .with_vector_store(
        Settings.CHROMA,
        persist_directory=persist_directory,
        collection_name=collection_name + "_bm25",
        search_type=Settings.SEARCH_BM25,
    )
    .with_llm(
        Settings.OLLAMA,
        model_name=model_name,
        system_prompt=Settings.DEFAULT_SYSTEM_PROMPT,
    )
    .build_rag(k=5)
)
rag_bm25.vector_store.ingest(data_path=data_path)
response = rag_bm25.generate("What classes are available in the vectorstore module?")
print(response)

# ── 3. Hybrid search (BM25 + semantic via RRF) ───────────────────────────────
print("\n=== Hybrid search (RRF) ===")
rag_hybrid = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings)
    .with_vector_store(
        Settings.CHROMA,
        persist_directory=persist_directory,
        collection_name=collection_name + "_hybrid",
        search_type=Settings.SEARCH_HYBRID,
        alpha=0.5,  # weight of semantic vs BM25 in the RRF merge (0=BM25 only, 1=semantic only)
    )
    .with_llm(
        Settings.OLLAMA,
        model_name=model_name,
        system_prompt=Settings.DEFAULT_SYSTEM_PROMPT,
    )
    .build_rag(k=5)
)
rag_hybrid.vector_store.ingest(data_path=data_path)
response = rag_hybrid.generate("Explain the Builder pattern used in RAGLight")
print(response)

# ── 4. Hybrid search via VectorStoreConfig (high-level API) ─────────────────
print("\n=== Hybrid search via RAGPipeline (high-level API) ===")
from raglight.rag.simple_rag_api import RAGPipeline
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.models.data_source_model import FolderSource

vector_store_config = VectorStoreConfig(
    embedding_model=Settings.DEFAULT_EMBEDDINGS_MODEL,
    provider=Settings.HUGGINGFACE,
    database=Settings.CHROMA,
    persist_directory=persist_directory,
    collection_name=collection_name + "_api",
    search_type=Settings.SEARCH_HYBRID,  # <-- hybrid mode
    hybrid_alpha=0.5,
)

config = RAGConfig(
    llm=model_name,
    provider=Settings.OLLAMA,
    k=5,
    knowledge_base=[FolderSource(path=data_path)],
)

pipeline = RAGPipeline(config, vector_store_config)
pipeline.build()
response = pipeline.generate("How does the ChromaVS vector store work?")
print(response)
