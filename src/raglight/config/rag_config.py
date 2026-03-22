from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..config.settings import Settings
from ..config.langfuse_config import LangfuseConfig
from ..cross_encoder.cross_encoder_model import CrossEncoderModel
from ..models.data_source_model import DataSource


@dataclass(kw_only=True)
class RAGConfig:
    cross_encoder_model: Optional[CrossEncoderModel] = None
    api_base: str = field(default=Settings.DEFAULT_OLLAMA_CLIENT)
    llm: str
    provider: str = field(default=Settings.OLLAMA)
    system_prompt: str = field(default=Settings.DEFAULT_SYSTEM_PROMPT)
    k: int = field(default=2)
    knowledge_base: Optional[List[DataSource]] = field(default=None)
    ignore_folders: list = field(
        default_factory=lambda: list(Settings.DEFAULT_IGNORE_FOLDERS)
    )
    langfuse_config: Optional[LangfuseConfig] = field(default=None)
    reformulation: bool = field(default=True)
    max_history: int = field(default=20)
