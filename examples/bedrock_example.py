"""
AWS Bedrock Example
===================
Prerequisites:
    pip install 'raglight[bedrock]'

AWS credentials (one of):
    - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    - AWS credentials file: ~/.aws/credentials
    - IAM role (EC2 / ECS / Lambda)

Supported models (examples):
    LLM:
        - anthropic.claude-3-5-sonnet-20241022-v2:0
        - anthropic.claude-3-haiku-20240307-v1:0
        - amazon.titan-text-express-v1
        - meta.llama3-8b-instruct-v1:0
    Embeddings:
        - amazon.titan-embed-text-v2:0
        - cohere.embed-english-v3
"""

from raglight.rag.simple_rag_api import RAGPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings
from raglight.config.rag_config import RAGConfig
from raglight.config.vector_store_config import VectorStoreConfig

Settings.setup_logging()

# --- Data sources ---
knowledge_base = [
    # FolderSource(path="data/my_docs"),
    GitHubSource(url="https://github.com/Bessouat40/RAGLight"),
]

# --- Vector store (embeddings via Bedrock Titan) ---
vector_store_config = VectorStoreConfig(
    provider=Settings.AWS_BEDROCK,
    embedding_model=Settings.AWS_BEDROCK_EMBEDDING_MODEL,  # amazon.titan-embed-text-v2:0
    database=Settings.CHROMA,
    persist_directory="./bedrockDb",
    collection_name="bedrock_collection",
)

# --- RAG pipeline (LLM via Bedrock Claude) ---
config = RAGConfig(
    provider=Settings.AWS_BEDROCK,
    llm=Settings.AWS_BEDROCK_LLM_MODEL,  # anthropic.claude-3-5-sonnet-20241022-v2:0
    knowledge_base=knowledge_base,
    # k=5,
    # system_prompt="You are a helpful assistant.",
)

pipeline = RAGPipeline(config, vector_store_config)
pipeline.build()

response = pipeline.generate(
    "How can I create a RAGPipeline using the raglight framework? Give me a Python example."
)
print(response)
