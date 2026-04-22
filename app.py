import hashlib
import os
import re

import streamlit as st

from rag_app import (
    answer_question,
    load_local_env,
    load_settings,
    prepare_rag_system,
    save_uploaded_pdf,
    validate_uploaded_file,
)


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# Load env
load_local_env()


@st.cache_resource(show_spinner=False)
def build_rag_state(file_bytes, file_name, cache_key):
    del cache_key
    suffix = os.path.splitext(file_name or "document.pdf")[1] or ".pdf"
    temp_pdf_path = save_uploaded_pdf(file_bytes, suffix=suffix)
    return prepare_rag_system(temp_pdf_path)


def get_file_signature(file_bytes, file_name):
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{file_name}:{digest}"


def clean_answer_text(answer: str) -> str:
    return re.sub(r"\n*\s*Sources:\s*page\s+\d+.*$", "", answer, flags=re.IGNORECASE).strip()


st.title("📄 Ask Questions About Your PDF")
st.caption("Upload a PDF and ask questions from it.")

# Warning for weak embedding
if os.getenv("EMBEDDING_PROVIDER", "").lower() == "local" and os.getenv(
    "LOCAL_EMBEDDING_MODEL", ""
).lower() == "hashing-v1":
    st.warning("Using a low-quality embedding model. Switch to MiniLM for better results.")

# Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if not uploaded_file:
    st.info("Upload a PDF to start.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_signature = get_file_signature(file_bytes, uploaded_file.name)

try:
    validate_uploaded_file(file_bytes, uploaded_file.name, load_settings())
except Exception as e:
    st.error(f"Invalid upload: {e}")
    st.stop()

# Build RAG
try:
    with st.spinner("Processing document..."):
        rag_state = build_rag_state(file_bytes, uploaded_file.name, file_signature)
except Exception as e:
    st.error(f"Error preparing document: {e}")
    st.stop()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_file" not in st.session_state:
    st.session_state.active_file = None

# Reset if new file
if st.session_state.active_file != file_signature:
    st.session_state.chat_history = []
    st.session_state.active_file = file_signature

# Input
question = st.text_input("Ask a question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Thinking..."):
                result = answer_question(
                    question.strip(),
                    rag_state,
                    chat_history=st.session_state.chat_history,
                )

            answer = result.get("answer", "Not available in document")
            answer = clean_answer_text(answer)
            sources = result.get("matches", [])

            st.session_state.chat_history.insert(
                0,
                {
                    "question": question.strip(),
                    "answer": answer,
                    "sources": sources[:1],
                },
            )

        except Exception as e:
            st.error(f"Error answering question: {e}")

# Display chat
for item in st.session_state.chat_history:
    st.subheader(item["question"])
    st.write(item["answer"])
