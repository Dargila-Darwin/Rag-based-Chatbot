from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
import faiss
import hashlib
import numpy as np
import os
import re
import sys
import tempfile
from collections import Counter


def load_local_env(env_path=".env"):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_client():
    if using_local_embeddings_only():
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_key:
            return OpenAI(
                api_key=openrouter_key,
                
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )

        if openai_key:
            return OpenAI(api_key=openai_key)

        raise ValueError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY so the chat model can answer questions."
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    if openai_key:
        return OpenAI(api_key=openai_key)

    raise ValueError(
        "Set OPENROUTER_API_KEY or OPENAI_API_KEY before running this script."
    )


def using_openrouter():
    return bool(os.getenv("OPENROUTER_API_KEY"))


def using_local_embeddings_only():
    return os.getenv("EMBEDDING_PROVIDER", "").strip().lower() == "local"


def get_models():
    if using_local_embeddings_only():
        embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        if using_openrouter():
            chat_model = os.getenv(
                "OPENROUTER_CHAT_MODEL",
                "openai/gpt-4o-mini",
            )
        else:
            chat_model = os.getenv(
                "OPENAI_CHAT_MODEL",
                "gpt-4o-mini",
            )
    elif using_openrouter():
        embedding_model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "openai/text-embedding-3-small",
        )
        chat_model = os.getenv(
            "OPENROUTER_CHAT_MODEL",
            "openai/gpt-4o-mini",
        )
    else:
        embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        chat_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-4o-mini",
        )

    # Catch a common config mistake: using an embedding model for chat.
    if "embed" in chat_model.lower():
        raise ValueError(
            f"Configured chat model looks like an embedding model: {chat_model}. "
            "Set OPENROUTER_CHAT_MODEL or OPENAI_CHAT_MODEL to a chat-capable model."
        )

    return embedding_model, chat_model


def format_api_error(exc):
    status_code = getattr(exc, "status_code", "unknown")
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    if body:
        return f"status={status_code}, body={body}"
    return f"status={status_code}"


def get_proxy_hint():
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
    configured = {
        name: value
        for name in proxy_vars
        if (value := os.getenv(name))
    }

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


def create_embeddings_in_batches(client, model, texts, batch_size=16):
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        print(
            f"Embedding batch {start // batch_size + 1}/"
            f"{(len(texts) + batch_size - 1) // batch_size}"
        )
        response = client.embeddings.create(**build_embedding_request(model, batch))
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def create_local_embeddings(local_model, texts, batch_size=16):
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        print(
            
            f"Embedding batch {start // batch_size + 1}/"
            f"{(len(texts) + batch_size - 1) // batch_size}"
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


def create_hash_embeddings(texts, dimension=512):
    embeddings = []

    for text in texts:
        vector = np.zeros(dimension, dtype="float32")
        tokens = re.findall(r"\w+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:8], "big") % dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[slot] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        embeddings.append(vector.tolist())

    return embeddings


def print_connection_help():
    print(
        "Could not connect to OpenRouter/OpenAI. Check your internet access and API base URL.",
        file=sys.stderr,
    )
    proxy_hint = get_proxy_hint()
    if proxy_hint:
        print(proxy_hint, file=sys.stderr)


def print_auth_help():
    print(
        "Authentication failed. Verify that OPENROUTER_API_KEY or OPENAI_API_KEY is valid "
        "for the selected provider.",
        file=sys.stderr,
    )


def print_output(text):
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        print(safe_text)


def extract_chat_text(response):
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
        f"Response id: {response_id}. "
        "Try a different chat model or inspect the raw provider response."
    )


def build_embedding_request(model, input_data, input_type=None):
    request = {
        "model": model,
        "input": input_data,
    }

    # NVIDIA's Nemotron embedding model expects an input_type hint.
    if "llama-nemotron-embed-vl" in model.lower():
        request["extra_body"] = {
            "input_type": input_type or "document",
        }

    return request


def load_and_split_document(pdf_path, chunk_size=500, chunk_overlap=100):
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


def normalize_text_tokens(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_query_token_counter(query):
    tokens = normalize_text_tokens(query)
    return Counter(token for token in tokens if len(token) > 2)


def is_heading_style_query(query):
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


def score_keyword_overlap(query_counter, document_text):
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


def rerank_matches(query, docs, indices, distances):
    query_counter = build_query_token_counter(query)
    rescored = []

    for index_position, doc_index in enumerate(indices[0]):
        if doc_index < 0 or doc_index >= len(docs):
            continue

        doc = docs[doc_index]
        distance = float(distances[0][index_position])
        semantic_score = 1.0 / (1.0 + max(distance, 0.0))
        keyword_score, overlap_count = score_keyword_overlap(
            query_counter=query_counter,
            document_text=doc.page_content,
        )
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
        key=lambda item: (
            item["combined_score"],
            item["keyword_score"],
            -item["distance"],
        ),
        reverse=True,
    )
    return rescored


def has_sufficient_context(query, scored_matches):
    if not scored_matches:
        return False

    best_match = scored_matches[0]
    if is_heading_style_query(query) and (
        query.strip().lower() in best_match["doc"].page_content.lower()
        or best_match["keyword_score"] >= 0.2
    ):
        return True

    query_tokens = build_query_token_counter(query)
    query_token_count = sum(query_tokens.values())

    if query_token_count == 0:
        return best_match["distance"] <= 1.1

    if best_match["keyword_score"] >= 0.45:
        return True

    if best_match["keyword_score"] >= 0.25 and best_match["distance"] <= 1.2:
        return True

    if (
        query_token_count <= 3
        and best_match["overlap_count"] >= 1
        and best_match["distance"] <= 0.95
    ):
        return True

    return False


def format_context_with_sources(scored_matches):
    context_blocks = []

    for item in scored_matches:
        doc = item["doc"]
        page_number = doc.metadata.get("page", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        context_blocks.append(
            f"[Page {page_number} | Chunk {chunk_id}]\n{doc.page_content}"
        )

    return "\n\n".join(context_blocks)


def format_source_list(scored_matches):
    pages = []
    for item in scored_matches:
        page_number = item["doc"].metadata.get("page")
        if page_number is None:
            continue
        page_label = str(page_number)
        if page_label not in pages:
            pages.append(page_label)

    if not pages:
        return "Sources: page unknown"

    return "Sources: " + ", ".join(f"page {page}" for page in pages)


def expand_heading_matches(query, docs, scored_matches, max_context_chunks=14):
    if not scored_matches:
        return []

    query_text = query.strip().lower()
    matched_positions = [
        index
        for index, doc in enumerate(docs)
        if query_text and query_text in doc.page_content.lower()
    ]

    if not matched_positions:
        best_chunk_id = scored_matches[0]["doc"].metadata.get("chunk_id")
        if best_chunk_id is None:
            return scored_matches[:max_context_chunks]
        matched_positions = [best_chunk_id - 1]

    start_index = min(
        position for position in matched_positions if 0 <= position < len(docs)
    )
    selected_positions = []

    for position in range(start_index, len(docs)):
        doc = docs[position]
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        first_line = lines[0].lower() if lines else ""

        if (
            position > start_index
            and first_line
            and first_line != query_text
            and len(first_line.split()) <= 8
        ):
            break

        selected_positions.append(position)
        if len(selected_positions) >= max_context_chunks:
            break

    expanded_matches = []
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


def create_document_embeddings(client, embedding_model, texts):
    if using_local_embeddings_only():
        if embedding_model == "hashing-v1":
            embeddings = create_hash_embeddings(
                texts,
                dimension=int(os.getenv("LOCAL_EMBEDDING_DIMENSION", "512")),
            )
            return embeddings, None

        SentenceTransformer = load_sentence_transformer()
        local_embedding_model = SentenceTransformer(embedding_model)
        embeddings = create_local_embeddings(
            local_model=local_embedding_model,
            texts=texts,
        )
        return embeddings, local_embedding_model

    embeddings = create_embeddings_in_batches(
        client=client,
        model=embedding_model,
        texts=texts,
    )
    return embeddings, None


def build_faiss_index(embeddings):
    if not embeddings:
        raise ValueError("The embedding API returned no vectors.")

    embedding_matrix = np.array(embeddings, dtype="float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index


def prepare_rag_system(pdf_path):
    load_local_env()
    client = build_client()
    embedding_model, chat_model = get_models()
    docs = load_and_split_document(
        pdf_path,
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "140")),
    )
    texts = [doc.page_content for doc in docs]

    try:
        embeddings, local_embedding_model = create_document_embeddings(
            client=client,
            embedding_model=embedding_model,
            texts=texts,
        )
    except Exception as exc:
        if using_local_embeddings_only():
            print(
                "Local embedding model failed to load or encode text.",
                file=sys.stderr,
            )
            print(
                "Use LOCAL_EMBEDDING_MODEL=hashing-v1 for a dependency-free fallback, "
                "or install sentence-transformers for higher-quality local embeddings.",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            raise

        if isinstance(exc, RateLimitError):
            print(
                "API quota exceeded while creating document embeddings.",
                file=sys.stderr,
            )
            raise
        if isinstance(exc, AuthenticationError):
            print_auth_help()
            print(
                "OpenRouter/OpenAI rejected the embedding request:",
                format_api_error(exc),
                file=sys.stderr,
            )
            raise
        if isinstance(exc, APIConnectionError):
            print_connection_help()
            print(f"Connection details: {exc}", file=sys.stderr)
            raise
        if isinstance(exc, ValueError):
            print(
                "The embedding request did not return vectors. "
                "This usually means the selected model does not support the embeddings API.",
                file=sys.stderr,
            )
            print(
                f"Embedding model in use: {embedding_model}",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            raise
        if isinstance(exc, APIStatusError):
            print(
                "OpenRouter/OpenAI rejected the embedding request:",
                format_api_error(exc),
                file=sys.stderr,
            )
            raise
        raise

    index = build_faiss_index(embeddings)
    return {
        "client": client,
        "embedding_model": embedding_model,
        "chat_model": chat_model,
        "docs": docs,
        "index": index,
        "local_embedding_model": local_embedding_model,
        "chunk_count": len(docs),
        "embedding_count": len(embeddings),
    }


def create_query_embedding(query, client, embedding_model, local_embedding_model):
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


def answer_question(query, rag_state, top_k=3, candidate_pool=8):
    client = rag_state["client"]
    embedding_model = rag_state["embedding_model"]
    chat_model = rag_state["chat_model"]
    docs = rag_state["docs"]
    index = rag_state["index"]
    local_embedding_model = rag_state["local_embedding_model"]
    heading_query = is_heading_style_query(query)

    try:
        query_embedding = create_query_embedding(
            query=query,
            client=client,
            embedding_model=embedding_model,
            local_embedding_model=local_embedding_model,
        )
    except Exception as exc:
        if using_local_embeddings_only():
            print("Local query embedding failed.", file=sys.stderr)
            print(f"Details: {exc}", file=sys.stderr)
            raise

        if isinstance(exc, RateLimitError):
            print("API quota exceeded while creating the query embedding.", file=sys.stderr)
            raise
        if isinstance(exc, AuthenticationError):
            print_auth_help()
            print(
                "OpenRouter/OpenAI rejected the query embedding request:",
                format_api_error(exc),
                file=sys.stderr,
            )
            raise
        if isinstance(exc, APIConnectionError):
            print_connection_help()
            print(f"Connection details: {exc}", file=sys.stderr)
            raise
        if isinstance(exc, ValueError):
            print(
                "The query embedding request did not return a vector. "
                "This usually means the selected model does not support the embeddings API.",
                file=sys.stderr,
            )
            print(
                f"Embedding model in use: {embedding_model}",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            raise
        if isinstance(exc, APIStatusError):
            print(
                "OpenRouter/OpenAI rejected the query embedding request:",
                format_api_error(exc),
                file=sys.stderr,
            )
            raise
        raise

    effective_top_k = 10 if heading_query else top_k
    effective_candidate_pool = 24 if heading_query else candidate_pool
    search_k = min(max(effective_top_k, effective_candidate_pool), len(docs))
    distances, indices = index.search(
        np.array([query_embedding], dtype="float32"),
        k=search_k,
    )
    scored_matches = rerank_matches(query, docs, indices, distances)[:effective_top_k]
    if heading_query:
        scored_matches = expand_heading_matches(
            query=query,
            docs=docs,
            scored_matches=scored_matches,
        )
    matched_docs = [item["doc"] for item in scored_matches]
    ranked_distances = [item["distance"] for item in scored_matches]
    context = format_context_with_sources(scored_matches)

    if not has_sufficient_context(query, scored_matches):
        return {
            "answer": "Not available in document",
            "distances": ranked_distances,
            "matches": matched_docs,
            "context": context,
        }

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the given context. "
                        "If the answer is missing, unclear, or only partially supported by the context, "
                        "reply exactly with 'Not available in document'. "
                        + (
                            "The user may be asking for a section heading. "
                            "When that happens, include all important points available under that heading in the provided context. "
                            "Use bullets when useful and do not shorten the answer too aggressively. "
                            if heading_query
                            else "When the answer is available, be concise. "
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
    except RateLimitError as exc:
        print("API quota exceeded while generating the answer.", file=sys.stderr)
        raise
    except AuthenticationError as exc:
        print_auth_help()
        print(
            "OpenRouter/OpenAI rejected the chat request:",
            format_api_error(exc),
            file=sys.stderr,
        )
        raise
    except APIConnectionError as exc:
        print_connection_help()
        print(f"Connection details: {exc}", file=sys.stderr)
        raise
    except APIStatusError as exc:
        print(
            "OpenRouter/OpenAI rejected the chat request:",
            format_api_error(exc),
            file=sys.stderr,
        )
        raise

    answer_text = extract_chat_text(response)
    return {
        "answer": answer_text if "Sources:" in answer_text else f"{answer_text}\n\n{format_source_list(scored_matches)}",
        "distances": ranked_distances,
        "matches": matched_docs,
        "context": context,
    }


def save_uploaded_pdf(uploaded_bytes, suffix=".pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_bytes)
        return temp_file.name


def main():
    try:
        rag_state = prepare_rag_system("taxreform.pdf")
    except Exception as exc:
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
            raise SystemExit(1) from exc

        print("Top match distances:", result["distances"])
        print_output(f"\nAnswer:\n {result['answer']}")


if __name__ == "__main__":
    main()
