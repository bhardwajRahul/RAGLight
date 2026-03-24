from __future__ import annotations
from typing import Optional, List
from typing_extensions import override

from langchain_openai import OpenAIEmbeddings

from ..config.settings import Settings
from .embeddings_model import EmbeddingsModel


class OpenAIEmbeddingsModel(EmbeddingsModel):
    def __init__(self, model_name: str, api_base: Optional[str] = None) -> None:
        resolved_api_base = api_base or Settings.DEFAULT_OPENAI_CLIENT
        super().__init__(model_name, api_base=resolved_api_base)

    @override
    def load(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.model_name,
            api_key=Settings.OPENAI_API_KEY,
            base_url=self.api_base,
        )

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)
