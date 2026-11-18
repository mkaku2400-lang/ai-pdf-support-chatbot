"""Quick end-to-end check of the document pipeline and the app, with no API key needed.

Usage:  python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from rag.demo import NOT_FOUND_MESSAGE, answer_from_document  # noqa: E402
from rag.pdf_loader import chunk_document, extract_pages  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402

# question -> a phrase the correct answer must contain
EXPECTED = {
    "How long does shipping take?": "3 to 5 business days",
    "What is your return policy?": "within 30 days",
    "What warranty do your products have?": "2-year limited warranty",
    "How can I contact customer support?": "1-800-555-0142",
    "when do I get my money back after a refund?": "5 to 7 business days",
    "do you deliver to Canada?": "7 to 14 business days",
    "what payment methods can I use": "PayPal",
    "can i cancel my order": "within 1 hour",
}
UNANSWERABLE = ["What is the capital of France?", "Do you sell car tires?"]


def check_pipeline() -> None:
    pages = extract_pages(str(ROOT / "sample_company_faq.pdf"))
    store = VectorStore(chunk_document(pages))
    print(f"Indexed {len(pages)} pages into {len(store.chunks)} chunks ({store.backend} index)")

    for question, expected in EXPECTED.items():
        hits = store.search(question)
        answer = answer_from_document(question, hits, store.embedder)
        assert expected in answer, f"{question!r}: expected {expected!r} in answer, got {answer!r}"
        print(f"  ok   {question}  (score {hits[0].score:.2f})")

    for question in UNANSWERABLE:
        answer = answer_from_document(question, store.search(question), store.embedder)
        assert answer == NOT_FOUND_MESSAGE, f"{question!r} should not be answered, got {answer!r}"
        print(f"  ok   {question}  -> correctly declined")


def check_app() -> None:
    from streamlit.testing.v1 import AppTest

    # Force Demo Mode regardless of what is in the local .env file.
    os.environ["ANTHROPIC_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["GEMINI_API_KEY"] = ""
    os.environ["GOOGLE_API_KEY"] = ""

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception, app.exception

    next(b for b in app.sidebar.button if "Sample" in b.label).click().run()
    assert not app.exception, app.exception
    assert app.session_state["store"] is not None, "sample PDF was not indexed"

    next(b for b in app.sidebar.button if b.label.startswith("How long")).click().run()
    assert not app.exception, app.exception
    reply = app.session_state["messages"][-1]
    assert reply["role"] == "assistant" and "3 to 5 business days" in reply["content"], reply
    assert reply["sources"], "answer should list its source passage"
    print("  ok   app starts, indexes the sample PDF and answers in Demo Mode")


if __name__ == "__main__":
    check_pipeline()
    check_app()
    print("All checks passed.")
