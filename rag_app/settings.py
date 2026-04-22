from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENV_PATH = ".env"


def load_local_env(env_path: str = DEFAULT_ENV_PATH) -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer. Received: {raw_value}") from exc

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    return value


@dataclass(frozen=True)
class AppSettings:
    app_data_dir: Path
    upload_dir: Path
    documents_dir: Path
    rag_chunk_size: int
    rag_chunk_overlap: int
    embedding_provider: str
    local_embedding_model: str
    local_embedding_dimension: int
    openrouter_base_url: str
    max_upload_size_mb: int
    answer_top_k: int
    answer_candidate_pool: int

    @property
    def using_local_embeddings(self) -> bool:
        return self.embedding_provider == "local"

    def ensure_directories(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)


def load_settings(env_path: str = DEFAULT_ENV_PATH) -> AppSettings:
    load_local_env(env_path)

    app_data_dir = Path(os.getenv("APP_DATA_DIR", "data")).resolve()
    upload_dir = Path(os.getenv("UPLOAD_DIR", str(app_data_dir / "uploads"))).resolve()
    documents_dir = Path(os.getenv("DOCUMENTS_DIR", str(app_data_dir / "documents"))).resolve()

    settings = AppSettings(
        app_data_dir=app_data_dir,
        upload_dir=upload_dir,
        documents_dir=documents_dir,
        rag_chunk_size=_read_int("RAG_CHUNK_SIZE", 700, minimum=100),
        rag_chunk_overlap=_read_int("RAG_CHUNK_OVERLAP", 140, minimum=0),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower(),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        local_embedding_dimension=_read_int("LOCAL_EMBEDDING_DIMENSION", 512, minimum=32),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).strip(),
        max_upload_size_mb=_read_int("MAX_UPLOAD_SIZE_MB", 25, minimum=1),
        answer_top_k=_read_int("ANSWER_TOP_K", 3, minimum=1),
        answer_candidate_pool=_read_int("ANSWER_CANDIDATE_POOL", 8, minimum=1),
    )

    if settings.rag_chunk_overlap >= settings.rag_chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")

    settings.ensure_directories()
    return settings
