from dataclasses import dataclass, field
from typing import Optional


@dataclass(kw_only=True)
class LangfuseConfig:
    """
    Configuration for Langfuse observability integration (v4.0.0).

    Langfuse traces every LangGraph node invocation (retrieve, rerank, generate)
    as a single trace per ``RAG.generate()`` call.

    Attributes:
        public_key (str): Langfuse public key (``LANGFUSE_PUBLIC_KEY``).
        secret_key (str): Langfuse secret key (``LANGFUSE_SECRET_KEY``).
        host (str): Langfuse server URL. Defaults to ``http://localhost:3000``.
        session_id (Optional[str]): Fixed trace/session ID to attach to every call
            made by this RAG instance. When ``None`` (default), a UUID hex is generated
            once at RAG construction time so all turns of the same conversation are
            grouped under the same trace.
            Must be 32 lowercase hex characters without dashes (Langfuse v4 requirement),
            e.g. ``uuid.uuid4().hex``.
    """

    public_key: str
    secret_key: str
    host: str = field(default="http://localhost:3000")
    session_id: Optional[str] = field(default=None)
