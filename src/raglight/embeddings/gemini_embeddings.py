from __future__ import annotations
from typing import Optional, List
from typing_extensions import override

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel


class GeminiEmbeddingsModel(EmbeddingsModel):
    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        super().__init__(model_name, api_base)

    @override
    def load(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=Settings.GEMINI_API_KEY,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
