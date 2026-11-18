"""AI Customer Support Agent - a RAG chatbot that answers from a company PDF.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag.config import get_settings
from rag.demo import GREETING_MESSAGE, MIN_CONFIDENT_SCORE, NOT_FOUND_MESSAGE, answer_from_document, stream_words
from rag.llm import LLMError, stream_answer
from rag.pdf_loader import PDFError, chunk_document, extract_pages
from rag.vector_store import VectorStore

SAMPLE_PDF = Path(__file__).parent / "sample_company_faq.pdf"
SAMPLE_QUESTIONS = [
    "How long does shipping take?",
    "What is your return policy?",
    "What warranty do your products have?",
    "How can I contact customer support?",
]
MAX_SOURCES_SHOWN = 3

st.set_page_config(page_title="AI Customer Support Agent", layout="centered")
settings = get_settings()

state = st.session_state
state.setdefault("messages", [])
state.setdefault("store", None)
state.setdefault("doc_info", None)
state.setdefault("last_upload_key", None)
state.setdefault("pending_question", None)
state.setdefault("doc_error", None)


def load_document(source, name: str) -> None:
    """Read, chunk and index a PDF, then start a fresh conversation about it."""
    state.doc_error = None
    try:
        with st.spinner(f"Reading and indexing {name}..."):
            pages = extract_pages(source)
            store = VectorStore(chunk_document(pages))
    except PDFError as exc:
        state.doc_error = str(exc)
        return
    state.store = store
    state.doc_info = {"name": name, "pages": len(pages), "chunks": len(store.chunks)}
    state.messages = [
        {
            "role": "assistant",
            "content": f"Hi! I've read **{name}**. Ask me anything about it.",
            "sources": [],
        }
    ]


def load_sample_document() -> None:
    if not SAMPLE_PDF.exists():
        try:
            from scripts.generate_sample_pdf import build_pdf

            build_pdf().output(str(SAMPLE_PDF))
        except Exception as exc:  # noqa: BLE001 - any failure here has the same fix
            state.doc_error = (
                f"The sample PDF is missing and could not be created ({exc}). "
                "Run: python scripts/generate_sample_pdf.py"
            )
            return
    load_document(str(SAMPLE_PDF), SAMPLE_PDF.name)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources from the document ({len(sources)})"):
        for source in sources:
            st.markdown(f"**Page {source['page']}**")
            st.caption(source["text"])


# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.subheader("Answer engine")

    has_api_key = settings.provider is not None
    demo_mode = st.toggle(
        "Demo Mode (no API key)",
        value=not has_api_key,
        disabled=not has_api_key,
        help="Answers are taken directly from the best matching passage, without calling an AI model.",
    )
    if demo_mode:
        st.caption("Demo Mode is on: answers are quoted from the document. Add an API key to `.env` for AI-written answers.")
    else:
        st.caption(f"AI Mode: {settings.provider.title()}, model `{settings.model}`.")

    st.divider()
    st.subheader("1. Add a company document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], help="FAQ, product catalog, policy manual...")
    if uploaded is not None:
        upload_key = (uploaded.name, uploaded.size)
        if upload_key != state.last_upload_key:
            state.last_upload_key = upload_key
            load_document(uploaded, uploaded.name)
    else:
        state.last_upload_key = None

    if st.button("Use Sample Company FAQ PDF", width="stretch"):
        load_sample_document()

    if state.doc_error:
        st.error(state.doc_error)
    if state.doc_info:
        info = state.doc_info
        st.success(f"**{info['name']}** is ready\n\n{info['pages']} pages · {info['chunks']} searchable sections")

    st.divider()
    st.subheader("2. Ask a question")
    st.caption("Type in the chat box, or try one of these with the sample PDF:")
    for sample_question in SAMPLE_QUESTIONS:
        if st.button(sample_question, width="stretch", disabled=state.store is None):
            state.pending_question = sample_question

    st.divider()
    if st.button("Clear chat", width="stretch", disabled=not state.messages):
        state.messages = state.messages[:1]
        st.rerun()

# -------------------------------------------------------------------- main --
st.title("AI Customer Support Agent")
st.caption("Answers customer questions from your own company document, and shows the passage each answer came from.")

if state.store is None:
    st.info(
        "No document loaded yet. Upload a PDF in the sidebar (an FAQ, product catalog or policy manual), "
        "or click **Use Sample Company FAQ PDF** to try it with a ready-made example."
    )

# A question sent while the previous answer was still being written interrupts that answer,
# which leaves a question with no reply. Drop it so the conversation stays in question/answer pairs.
if state.messages and state.messages[-1]["role"] == "user":
    state.messages.pop()

for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

typed_question = st.chat_input("Ask a question about the document...", disabled=state.store is None)
question = typed_question or state.pending_question
state.pending_question = None

if question and state.store is not None:
    history = list(state.messages)
    state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = state.store.search(question, k=settings.top_k)
    sources = [
        {"page": hit.chunk.page, "text": hit.chunk.text}
        for hit in hits[:MAX_SOURCES_SHOWN]
        if hit.score >= MIN_CONFIDENT_SCORE
    ]

    with st.chat_message("assistant"):
        answer = None
        if demo_mode:
            demo_answer = answer_from_document(question, hits, state.store.embedder)
            if demo_answer in (NOT_FOUND_MESSAGE, GREETING_MESSAGE):
                sources = []
            else:
                sources = sources[:1]  # the answer is quoted from the top passage
            answer = st.write_stream(stream_words(demo_answer))
        else:
            try:
                answer = st.write_stream(stream_answer(settings, question, hits, history))
            except LLMError as exc:
                st.error(str(exc))
        if answer:
            show_sources(sources)
            state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        else:
            # Keep the history valid: drop the question that got no answer.
            state.messages.pop()
