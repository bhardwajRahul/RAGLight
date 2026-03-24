"""
Qdrant vector store example — works on Windows without a C++ compiler.

Install:
    pip install "raglight[qdrant]"

Local mode (on-disk):
    No server required. Data is stored in `persist_directory`.

Remote mode (HTTP):
    Start a Qdrant instance, e.g. with Docker:
        docker run -p 6333:6333 qdrant/qdrant
    Then pass host/port instead of persist_directory.
"""

import uuid
from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()
Settings.setup_logging()

# --- Configuration -----------------------------------------------------------

persist_directory = "./qdrantDb"       # local mode — change to None + set host/port for remote
# host = "localhost"                   # remote mode
# port = 6333                          # remote mode (default)

model_embeddings = Settings.DEFAULT_EMBEDDINGS_MODEL  # "all-MiniLM-L6-v2"
model_name = "llama3.1:8b"
collection_name = str(uuid.uuid4())

# --- Build the pipeline (local mode) -----------------------------------------

rag = (
    Builder()
    .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings)
    .with_vector_store(
        Settings.QDRANT,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    .with_llm(
        Settings.OLLAMA,
        model_name=model_name,
        system_prompt=Settings.DEFAULT_SYSTEM_PROMPT,
        options={"num_ctx": 8192},
    )
    .build_rag(k=5)
)

# --- Remote mode example (commented out) -------------------------------------
#
# rag = (
#     Builder()
#     .with_embeddings(Settings.HUGGINGFACE, model_name=model_embeddings)
#     .with_vector_store(
#         Settings.QDRANT,
#         host=host,
#         port=port,
#         collection_name=collection_name,
#     )
#     .with_llm(Settings.OLLAMA, model_name=model_name)
#     .build_rag(k=5)
# )

# --- Ingest & chat -----------------------------------------------------------

rag.vector_store.ingest(data_path="./src/raglight")

while True:
    query = input(">>> ")
    if query in ("quit", "bye"):
        print("See you soon!")
        break
    response = rag.generate(query)
    print(response)
