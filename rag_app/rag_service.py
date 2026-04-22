from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from rag_app.settings import AppSettings, load_local_env, load_settings


logger = logging.getLogger(__name__)


def using_openrouter() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def using_local_embeddings_only() -> bool:
    return os.getenv("EMBEDDING_PROVIDER", "").strip().lower() == "local"


def build_client() -> OpenAI:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    if openai_key:
        return OpenAI(api_key=openai_key)

    if using_local_embeddings_only():
        raise ValueError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY so the chat model can answer questions."
        )

    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY before starting the app.")


def get_models(settings: AppSettings | None = None) -> tuple[str, str]:
    settings = settings or load_settings()

    if settings.using_local_embeddings:
        embedding_model = settings.local_embedding_model
        chat_model = (
            os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
            if using_openrouter()
            else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        )
    elif using_openrouter():
        embedding_model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "openai/text-embedding-3-small",
        )
        chat_model = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
    else:
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    if "embed" in chat_model.lower():
        raise ValueError(
            f"Configured chat model looks like an embedding model: {chat_model}. "
            "Set OPENROUTER_CHAT_MODEL or OPENAI_CHAT_MODEL to a chat-capable model."
        )

    return embedding_model, chat_model


def format_api_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", "unknown")
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    return f"status={status_code}, body={body}" if body else f"status={status_code}"


def get_proxy_hint() -> str | None:
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
    configured = {name: value for name in proxy_vars if (value := os.getenv(name))}

    if not configured:
        return None

    bad_loopback_ports = ("127.0.0.1:9", "localhost:9")
    suspicious = [
        f"{name}={value}"
        for name, value in configured.items()
        if any(port in value for port in bad_loopback_ports)
    ]

    if suspicious:
        return (
            "Suspicious proxy settings detected: "
            + ", ".join(suspicious)
            + ". Clear these environment variables or set them to a working proxy."
        )

    return (
        "Proxy settings are configured: "
        + ", ".join(f"{name}={value}" for name, value in configured.items())
        + ". If API calls fail to connect, verify that this proxy is reachable."
    )


def build_embedding_request(model: str, input_data: list[str], input_type: str | None = None) -> dict[str, Any]:
    request: dict[str, Any] = {"model": model, "input": input_data}
    if "llama-nemotron-embed-vl" in model.lower():
        request["extra_body"] = {"input_type": input_type or "document"}
    return request


def create_embeddings_in_batches(
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int = 16,
) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.info(
            "Embedding batch %s/%s",
            start // batch_size + 1,
            (len(texts) + batch_size - 1) // batch_size,
        )
        response = client.embeddings.create(**build_embedding_request(model, batch))
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def create_local_embeddings(local_model: Any, texts: list[str], batch_size: int = 16) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.info(
            "Embedding batch %s/%s",
            start // batch_size + 1,
            (len(texts) + batch_size - 1) // batch_size,
        )
        batch_embeddings = local_model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings.extend(batch_embeddings.tolist())

    return embeddings


def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed for the selected local model."
        ) from exc
    return SentenceTransformer


def resolve_local_embedding_backend(embedding_model: str) -> tuple[str, Any]:
    if embedding_model == "hashing-v1":
        return embedding_model, None

    try:
        SentenceTransformer = load_sentence_transformer()
    except ImportError:
        fallback_model = "hashing-v1"
        os.environ["LOCAL_EMBEDDING_MODEL"] = fallback_model
        logger.warning(
            "sentence-transformers is unavailable; falling back to %s local embeddings.",
            fallback_model,
        )
        return fallback_model, None

    return embedding_model, SentenceTransformer(embedding_model)


def create_hash_embeddings(texts: list[str], dimension: int = 512) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for text in texts:
        vector = np.zeros(dimension, dtype="float32")
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:8], "big") % dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[slot] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        embeddings.append(vector.tolist())

    return embeddings


def print_connection_help() -> None:
    print(
        "Could not connect to OpenRouter/OpenAI. Check your internet access and API base URL.",
        file=sys.stderr,
    )
    proxy_hint = get_proxy_hint()
    if proxy_hint:
        print(proxy_hint, file=sys.stderr)


def print_auth_help() -> None:
    print(
        "Authentication failed. Verify that OPENROUTER_API_KEY or OPENAI_API_KEY is valid "
        "for the selected provider.",
        file=sys.stderr,
    )


def print_output(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def extract_chat_text(response: Any) -> str:
    if response is None:
        raise ValueError("Chat API returned no response object.")

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value)

            combined = "\n".join(part.strip() for part in text_parts if part.strip())
            if combined:
                return combined

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    response_id = getattr(response, "id", "unknown")
    raise ValueError(
        "The chat model returned a response without answer text. "
        f"Response id: {response_id}. Try a different chat model or inspect the raw provider response."
    )


def load_and_split_document(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.split_documents(documents)

    if not docs:
        raise ValueError(f"No text chunks were created from {pdf_path}.")

    for chunk_id, doc in enumerate(docs, start=1):
        doc.metadata["chunk_id"] = chunk_id

    return docs


def load_documents_from_directory(
    documents_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    supported_patterns = ("*.pdf", "*.txt", "*.md")
    file_paths: list[Path] = []
    for pattern in supported_patterns:
        file_paths.extend(sorted(documents_dir.rglob(pattern)))

    if not file_paths:
        raise ValueError(
            f"No supported documents were found in {documents_dir}. "
            "Add .pdf, .txt, or .md files to the directory."
        )

    docs = []
    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loaded_docs = PyMuPDFLoader(str(file_path)).load()
        else:
            loaded_docs = TextLoader(str(file_path), encoding="utf-8").load()

        for doc in loaded_docs:
            doc.metadata["source_path"] = str(file_path)
            docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        raise ValueError(f"No text chunks were created from documents in {documents_dir}.")

    for chunk_id, doc in enumerate(split_docs, start=1):
        doc.metadata["chunk_id"] = chunk_id

    return split_docs


def normalize_text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize_lookup_text(text: str) -> str:
    return " ".join(normalize_text_tokens(text))


def extract_heading_reference(query: str, docs: list[Any]) -> str | None:
    normalized_query = normalize_lookup_text(query)
    query_tokens = normalized_query.split()
    if len(query_tokens) < 2:
        return None

    normalized_docs = [normalize_lookup_text(doc.page_content) for doc in docs]
    best_match: str | None = None
    best_score = 0

    for span in range(min(8, len(query_tokens)), 1, -1):
        for start in range(0, len(query_tokens) - span + 1):
            candidate_tokens = query_tokens[start : start + span]
            candidate = " ".join(candidate_tokens)
            if candidate in {"main topic", "topic of", "of the", "the document"}:
                continue

            if any(candidate in doc_text for doc_text in normalized_docs):
                score = span
                if candidate.startswith("part "):
                    score += 3
                if "analysis" in candidate_tokens:
                    score += 2
                if score > best_score:
                    best_match = candidate
                    best_score = score

        if best_match:
            break

    return best_match


def build_query_token_counter(query: str) -> Counter[str]:
    tokens = normalize_text_tokens(query)
    return Counter(token for token in tokens if len(token) > 2)


def is_heading_style_query(query: str) -> bool:
    stripped_query = query.strip()
    if not stripped_query:
        return False
    if any(mark in stripped_query for mark in ("?", "\n", ":")):
        return False
    words = stripped_query.split()
    if len(words) > 8:
        return False
    lowered = stripped_query.lower()
    prompt_like_starters = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "tell me",
        "give me",
        "explain",
        "summarize",
        "list",
    )
    return not lowered.startswith(prompt_like_starters)


def looks_like_heading_reference(query: str, docs: list[Any]) -> bool:
    return is_heading_style_query(query) or bool(extract_heading_reference(query, docs))


def score_keyword_overlap(query_counter: Counter[str], document_text: str) -> tuple[float, int]:
    if not query_counter:
        return 0.0, 0

    document_counter = Counter(normalize_text_tokens(document_text))
    overlap = sum(
        min(count, document_counter.get(token, 0))
        for token, count in query_counter.items()
    )
    total = sum(query_counter.values())
    if total == 0:
        return 0.0, 0
    return overlap / total, overlap


def rerank_matches(query: str, docs: list[Any], indices: Any, distances: Any) -> list[dict[str, Any]]:
    query_counter = build_query_token_counter(query)
    rescored: list[dict[str, Any]] = []

    for index_position, doc_index in enumerate(indices[0]):
        if doc_index < 0 or doc_index >= len(docs):
            continue

        doc = docs[doc_index]
        distance = float(distances[0][index_position])
        semantic_score = 1.0 / (1.0 + max(distance, 0.0))
        keyword_score, overlap_count = score_keyword_overlap(query_counter, doc.page_content)
        phrase_bonus = 0.08 if query.strip().lower() in doc.page_content.lower() else 0.0
        combined_score = (semantic_score * 0.65) + (keyword_score * 0.35) + phrase_bonus
        rescored.append(
            {
                "doc": doc,
                "distance": distance,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "overlap_count": overlap_count,
                "combined_score": combined_score,
            }
        )

    rescored.sort(
        key=lambda item: (item["combined_score"], item["keyword_score"], -item["distance"]),
        reverse=True,
    )
    return rescored


def has_sufficient_context(query: str, scored_matches: list[dict[str, Any]]) -> bool:
    if not scored_matches:
        return False

    best_match = scored_matches[0]
    if best_match["distance"] <= 0.55:
        return True
    if is_heading_style_query(query) and (
        query.strip().lower() in best_match["doc"].page_content.lower()
        or best_match["keyword_score"] >= 0.15
        or best_match["overlap_count"] >= 2
    ):
        return True

    query_tokens = build_query_token_counter(query)
    query_token_count = sum(query_tokens.values())

    if query_token_count == 0:
        return best_match["distance"] <= 1.35
    if best_match["keyword_score"] >= 0.3:
        return True
    if best_match["keyword_score"] >= 0.18 and best_match["distance"] <= 1.35:
        return True
    if best_match["overlap_count"] >= max(2, min(4, query_token_count)) and best_match["distance"] <= 1.5:
        return True
    if query_token_count <= 3 and best_match["overlap_count"] >= 1 and best_match["distance"] <= 1.2:
        return True
    return False


def format_context_with_sources(scored_matches: list[dict[str, Any]]) -> str:
    blocks = []
    for item in scored_matches:
        doc = item["doc"]
        page_number = doc.metadata.get("page", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source_name = Path(doc.metadata.get("source_path", "unknown")).name
        blocks.append(
            f"[Source {source_name} | Page {page_number} | Chunk {chunk_id}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def format_source_list(scored_matches: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for item in scored_matches:
        source_name = Path(item["doc"].metadata.get("source_path", "unknown")).name
        page_number = item["doc"].metadata.get("page")
        label = source_name if page_number is None else f"{source_name} page {page_number}"
        if label not in labels:
            labels.append(label)
    if not labels:
        return "Sources: unknown"
    return "Sources: " + ", ".join(labels)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        normalized = normalize_whitespace(line).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_lines.append(normalize_whitespace(line))
    return unique_lines


def get_section_text_blocks(compact_text: str) -> dict[str, str]:
    section_patterns = {
        "objectives": r"Objectives\s*:?\s*(.*?)(?=\bKey\s+Policy\s+Reforms\b|\bExpected\s+Impact\b|$)",
        "key policy reforms": r"Key\s+Policy\s+Reforms\s*:?\s*(.*?)(?=\bExpected\s+Impact\b|$)",
        "expected impact": r"Expected\s+Impact\s*:?\s*(.*?)(?=\b\d+\s+[A-Z][A-Za-z/&\-\s]{3,}\b|$)",
    }
    blocks: dict[str, str] = {}
    for name, pattern in section_patterns.items():
        match = re.search(pattern, compact_text, flags=re.IGNORECASE)
        if match:
            section_text = match.group(1).strip(" :-")
            if section_text:
                blocks[name] = section_text
    return blocks


def format_bullet_block(title: str, raw_section: str) -> str:
    if title.lower() == "key policy reforms":
        items = parse_policy_reform_items(raw_section)
        if items:
            formatted_items = "\n".join(f"- {item_title}: {item_body}" for item_title, item_body in items)
            return f"{title}:\n{formatted_items}"

    bullet_candidates = [
        part.strip(" .:-")
        for part in re.split(r"\s*[?•➢]\s+|\s+[?•➢]\s*|[?•➢]\s*", raw_section)
        if part.strip(" .:-")
    ]
    bullet_lines = deduplicate_lines(bullet_candidates)
    if not bullet_lines:
        return f"{title}: {raw_section}"
    formatted_bullets = "\n".join(f"- {line}" for line in bullet_lines)
    return f"{title}:\n{formatted_bullets}"


def parse_policy_reform_items(raw_section: str) -> list[tuple[str, str]]:
    compact = normalize_whitespace(raw_section)
    pattern = re.compile(
        r"([A-Z][A-Za-z/&\-]*(?:\s+[A-Za-z/&\-]+){0,8})\s*[?•➢]\s*(.*?)(?="
        r"(?:[A-Z][A-Za-z/&\-]*(?:\s+[A-Za-z/&\-]+){0,8})\s*[?•➢]|$)"
    )
    items: list[tuple[str, str]] = []
    for title, body in pattern.findall(compact):
        clean_title = normalize_whitespace(title)
        clean_body = normalize_whitespace(body).strip(" .:-")
        if clean_title and clean_body:
            items.append((clean_title, clean_body))
    return items


def extract_subsection_answer(query: str, scored_matches: list[dict[str, Any]]) -> str | None:
    if not scored_matches:
        return None

    query_text = normalize_lookup_text(query)
    combined_text = "\n".join(item["doc"].page_content for item in scored_matches)
    compact_text = normalize_whitespace(combined_text)
    section_blocks = get_section_text_blocks(compact_text)

    key_policy_block = section_blocks.get("key policy reforms")
    if key_policy_block:
        for title, body in parse_policy_reform_items(key_policy_block):
            normalized_title = normalize_lookup_text(title)
            if normalized_title and normalized_title in query_text:
                return f"{title}:\n- {body}"

    requested_section: str | None = None
    for label in ("expected impact", "objectives", "key policy reforms"):
        if label in query_text:
            requested_section = label
            break

    if requested_section is None:
        return None

    raw_section = section_blocks.get(requested_section)
    if not raw_section:
        return None

    heading = requested_section.title()
    return format_bullet_block(heading, raw_section)


def is_probable_section_heading(line: str, query_text: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    normalized_line = normalize_lookup_text(stripped)
    if normalized_line == normalize_lookup_text(query_text):
        return False

    if re.match(r"^\d+\s+", stripped):
        return True

    words = stripped.split()
    if len(words) > 10:
        return False

    if stripped[0].islower():
        return False

    capitalized_words = sum(
        1 for word in words if word[:1].isupper() or "/" in word or "&" in word
    )
    return capitalized_words >= max(2, len(words) - 1)


def expand_heading_matches(
    query: str,
    docs: list[Any],
    scored_matches: list[dict[str, Any]],
    max_context_chunks: int = 14,
) -> list[dict[str, Any]]:
    if not scored_matches:
        return []

    query_text = query.strip().lower()
    matched_positions = [
        index for index, doc in enumerate(docs) if query_text and query_text in doc.page_content.lower()
    ]

    if not matched_positions:
        best_chunk_id = scored_matches[0]["doc"].metadata.get("chunk_id")
        if best_chunk_id is None:
            return scored_matches[:max_context_chunks]
        matched_positions = [best_chunk_id - 1]

    start_index = min(position for position in matched_positions if 0 <= position < len(docs))
    selected_positions = []

    for position in range(start_index, len(docs)):
        doc = docs[position]
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        first_line = lines[0].lower() if lines else ""

        if position > start_index and first_line and is_probable_section_heading(lines[0], query_text):
            break

        selected_positions.append(position)
        if len(selected_positions) >= max_context_chunks:
            break

    expanded_matches: list[dict[str, Any]] = []
    for position in selected_positions:
        doc = docs[position]
        existing_item = next(
            (
                item
                for item in scored_matches
                if item["doc"].metadata.get("chunk_id") == doc.metadata.get("chunk_id")
            ),
            None,
        )
        expanded_matches.append(
            existing_item
            or {
                "doc": doc,
                "distance": 0.0,
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "overlap_count": 0,
                "combined_score": 0.0,
            }
        )
    return expanded_matches


def create_document_embeddings(
    client: OpenAI,
    embedding_model: str,
    texts: list[str],
) -> tuple[list[list[float]], Any, str]:
    if using_local_embeddings_only():
        resolved_model, local_embedding_model = resolve_local_embedding_backend(embedding_model)
        if resolved_model == "hashing-v1":
            return create_hash_embeddings(
                texts,
                dimension=int(os.getenv("LOCAL_EMBEDDING_DIMENSION", "512")),
            ), None, resolved_model

        return create_local_embeddings(local_embedding_model, texts), local_embedding_model, resolved_model

    return create_embeddings_in_batches(client=client, model=embedding_model, texts=texts), None, embedding_model


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    if not embeddings:
        raise ValueError("The embedding API returned no vectors.")

    embedding_matrix = np.array(embeddings, dtype="float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index


def prepare_rag_system(pdf_path: str, settings: AppSettings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    load_local_env()
    client = build_client()
    embedding_model, chat_model = get_models(settings)
    docs = load_and_split_document(
        pdf_path,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    texts = [doc.page_content for doc in docs]

    try:
        embeddings, local_embedding_model, resolved_embedding_model = create_document_embeddings(
            client=client,
            embedding_model=embedding_model,
            texts=texts,
        )
    except Exception as exc:
        if settings.using_local_embeddings:
            logger.exception("Local embedding model failed.")
            print(
                "Use LOCAL_EMBEDDING_MODEL=hashing-v1 for a dependency-free fallback, "
                "or install sentence-transformers for higher-quality local embeddings.",
                file=sys.stderr,
            )
            raise

        if isinstance(exc, RateLimitError):
            logger.exception("API quota exceeded while creating embeddings.")
            raise
        if isinstance(exc, AuthenticationError):
            print_auth_help()
            print("OpenRouter/OpenAI rejected the embedding request:", format_api_error(exc), file=sys.stderr)
            raise
        if isinstance(exc, APIConnectionError):
            print_connection_help()
            print(f"Connection details: {exc}", file=sys.stderr)
            raise
        if isinstance(exc, ValueError):
            print(
                "The embedding request did not return vectors. This usually means the selected model does not support the embeddings API.",
                file=sys.stderr,
            )
            print(f"Embedding model in use: {embedding_model}", file=sys.stderr)
            raise
        if isinstance(exc, APIStatusError):
            print("OpenRouter/OpenAI rejected the embedding request:", format_api_error(exc), file=sys.stderr)
            raise
        raise

    return {
        "client": client,
        "embedding_model": resolved_embedding_model,
        "chat_model": chat_model,
        "docs": docs,
        "index": build_faiss_index(embeddings),
        "local_embedding_model": local_embedding_model,
        "chunk_count": len(docs),
        "embedding_count": len(embeddings),
    }


def prepare_rag_system_from_directory(settings: AppSettings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    load_local_env()
    client = build_client()
    embedding_model, chat_model = get_models(settings)
    docs = load_documents_from_directory(
        settings.documents_dir,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    texts = [doc.page_content for doc in docs]
    embeddings, local_embedding_model, resolved_embedding_model = create_document_embeddings(
        client=client,
        embedding_model=embedding_model,
        texts=texts,
    )

    sources = sorted(
        {
            Path(doc.metadata.get("source_path", "unknown")).name
            for doc in docs
        }
    )
    return {
        "client": client,
        "embedding_model": resolved_embedding_model,
        "chat_model": chat_model,
        "docs": docs,
        "index": build_faiss_index(embeddings),
        "local_embedding_model": local_embedding_model,
        "chunk_count": len(docs),
        "embedding_count": len(embeddings),
        "source_count": len(sources),
        "sources": sources,
        "documents_dir": str(settings.documents_dir),
    }


def create_query_embedding(
    query: str,
    client: OpenAI,
    embedding_model: str,
    local_embedding_model: Any,
) -> list[float]:
    if using_local_embeddings_only():
        if embedding_model == "hashing-v1":
            return create_hash_embeddings(
                [query],
                dimension=int(os.getenv("LOCAL_EMBEDDING_DIMENSION", "512")),
            )[0]

        if local_embedding_model is None:
            raise ValueError("Local embedding model is not loaded.")

        return local_embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()

    response = client.embeddings.create(
        **build_embedding_request(
            embedding_model,
            [query],
            input_type="query",
        )
    )
    return response.data[0].embedding


def answer_question(
    query: str,
    rag_state: dict[str, Any],
    chat_history: list[dict[str, str]] | None = None,
    top_k: int | None = None,
    candidate_pool: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    top_k = top_k or settings.answer_top_k
    candidate_pool = candidate_pool or settings.answer_candidate_pool

    client = rag_state["client"]
    embedding_model = rag_state["embedding_model"]
    chat_model = rag_state["chat_model"]
    docs = rag_state["docs"]
    index = rag_state["index"]
    local_embedding_model = rag_state["local_embedding_model"]
    heading_reference = extract_heading_reference(query, docs)
    retrieval_query = heading_reference or query
    heading_query = looks_like_heading_reference(query, docs)

    try:
        query_embedding = create_query_embedding(
            retrieval_query,
            client,
            embedding_model,
            local_embedding_model,
        )
    except Exception as exc:
        if using_local_embeddings_only():
            logger.exception("Local query embedding failed.")
            raise
        if isinstance(exc, RateLimitError):
            print("API quota exceeded while creating the query embedding.", file=sys.stderr)
            raise
        if isinstance(exc, AuthenticationError):
            print_auth_help()
            print("OpenRouter/OpenAI rejected the query embedding request:", format_api_error(exc), file=sys.stderr)
            raise
        if isinstance(exc, APIConnectionError):
            print_connection_help()
            print(f"Connection details: {exc}", file=sys.stderr)
            raise
        if isinstance(exc, ValueError):
            print(
                "The query embedding request did not return a vector. This usually means the selected model does not support the embeddings API.",
                file=sys.stderr,
            )
            print(f"Embedding model in use: {embedding_model}", file=sys.stderr)
            raise
        if isinstance(exc, APIStatusError):
            print("OpenRouter/OpenAI rejected the query embedding request:", format_api_error(exc), file=sys.stderr)
            raise
        raise

    effective_top_k = 10 if heading_query else top_k
    effective_candidate_pool = 24 if heading_query else candidate_pool
    search_k = min(max(effective_top_k, effective_candidate_pool), len(docs))
    distances, indices = index.search(np.array([query_embedding], dtype="float32"), k=search_k)
    scored_matches = rerank_matches(retrieval_query, docs, indices, distances)[:effective_top_k]
    if heading_query:
        scored_matches = expand_heading_matches(
            query=retrieval_query,
            docs=docs,
            scored_matches=scored_matches,
        )

    matched_docs = [item["doc"] for item in scored_matches]
    ranked_distances = [item["distance"] for item in scored_matches]
    context = format_context_with_sources(scored_matches)
    best_match = scored_matches[0] if scored_matches else None
    subsection_answer = extract_subsection_answer(query, scored_matches)

    if not has_sufficient_context(retrieval_query, scored_matches):
        if best_match is not None:
            fallback_answer = (
                "I found the closest matching passage in the document, but confidence is low.\n\n"
                f"{best_match['doc'].page_content.strip()}"
            )
        else:
            fallback_answer = "Not available in document"
        return {
            "answer": fallback_answer,
            "distances": ranked_distances,
            "matches": matched_docs,
            "context": context,
            "sources": format_source_list(scored_matches),
        }

    if subsection_answer:
        return {
            "answer": subsection_answer,
            "distances": ranked_distances,
            "matches": matched_docs,
            "context": context,
            "sources": format_source_list(scored_matches),
        }

    history_blocks = []
    for item in (chat_history or [])[-6:]:
        user_question = item.get("question", "").strip()
        assistant_answer = item.get("answer", "").strip()
        if user_question and assistant_answer:
            history_blocks.append(
                f"User: {user_question}\nAssistant: {assistant_answer}"
            )
    conversation_context = "\n\n".join(history_blocks) or "No previous conversation."

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the given context. "
                        "Use the chat history only to resolve follow-up references such as pronouns or omitted subjects. "
                        "If the answer is missing, unclear, or only partially supported by the context, "
                        "reply exactly with 'Not available in document'. "
                        + (
                            "The user may be asking about a section heading or a heading embedded in a longer question. When that happens, answer from the section content under that heading, not from the table of contents. Include the main topic and important points available under that heading in the provided context. Use bullets when useful and do not shorten the answer too aggressively. "
                            if heading_query
                            else "When the answer is available, be concise. "
                        )
                        + "Do not include source lines or page citations in the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Chat history:\n{conversation_context}\n\n"
                        f"Context:\n{context}\n\nQuestion: {query}"
                    ),
                },
            ],
        )
    except RateLimitError:
        print("API quota exceeded while generating the answer.", file=sys.stderr)
        raise
    except AuthenticationError as exc:
        print_auth_help()
        print("OpenRouter/OpenAI rejected the chat request:", format_api_error(exc), file=sys.stderr)
        raise
    except APIConnectionError as exc:
        print_connection_help()
        print(f"Connection details: {exc}", file=sys.stderr)
        raise
    except APIStatusError as exc:
        print("OpenRouter/OpenAI rejected the chat request:", format_api_error(exc), file=sys.stderr)
        raise

    answer_text = extract_chat_text(response)
    sources = format_source_list(scored_matches)
    return {
        "answer": answer_text,
        "distances": ranked_distances,
        "matches": matched_docs,
        "context": context,
        "sources": sources,
    }


def save_uploaded_pdf(uploaded_bytes: bytes, suffix: str = ".pdf", settings: AppSettings | None = None) -> str:
    settings = settings or load_settings()
    suffix = suffix if suffix.lower() == ".pdf" else ".pdf"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=settings.upload_dir,
    ) as temp_file:
        temp_file.write(uploaded_bytes)
        return temp_file.name


def validate_uploaded_file(file_bytes: bytes, file_name: str, settings: AppSettings) -> None:
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    size_in_mb = len(file_bytes) / (1024 * 1024)
    if size_in_mb > settings.max_upload_size_mb:
        raise ValueError(
            f"File size exceeds the configured limit of {settings.max_upload_size_mb} MB."
        )

    if not file_bytes.startswith(b"%PDF"):
        raise ValueError("Uploaded file does not look like a valid PDF.")


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        rag_state = prepare_rag_system(str(Path("taxreform.pdf").resolve()))
    except Exception as exc:
        logger.exception("Application startup failed: %s", exc)
        raise SystemExit(1) from exc

    print("Total chunks:", rag_state["chunk_count"])
    print("Embedding model:", rag_state["embedding_model"])
    print("Chat model:", rag_state["chat_model"])
    print("Embeddings created:", rag_state["embedding_count"])
    if using_local_embeddings_only():
        print("Embedding provider: local")

    while True:
        query = input("\nAsk a question about the document (or type 'exit'): ").strip()
        if not query:
            print("Please enter a question.")
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting document Q&A.")
            break

        try:
            result = answer_question(query, rag_state)
        except Exception as exc:
            logger.exception("Failed to answer question: %s", exc)
            raise SystemExit(1) from exc

        print("Top match distances:", result["distances"])
        print_output(f"\nAnswer:\n {result['answer']}")


if __name__ == "__main__":
    main()
