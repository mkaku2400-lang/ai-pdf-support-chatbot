"""Demo Mode: answer without any AI API key.

Retrieval is real (the same search the AI mode uses). The answer is then taken
word-for-word from the best matching passage instead of being written by an
AI model. Good for demos and for testing a document before paying for an API.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from .embeddings import HashedTfidfEmbedder
from .pdf_loader import ANSWER_MARKER, split_sentences
from .vector_store import SearchHit

# Below this similarity the passage is not a trustworthy match for the question.
MIN_CONFIDENT_SCORE = 0.12

NOT_FOUND_MESSAGE = (
    "I'm sorry, I couldn't find that in the company document. "
    "Please try rephrasing your question, or contact our support team for help."
)
GREETING_MESSAGE = (
    "Hello! I'm the support assistant. Ask me anything about the uploaded "
    "company document, for example shipping, returns or warranty."
)

_GREETING = re.compile(r"^\s*(hi|hello|hey|salam|assalam\w*|good (morning|afternoon|evening))\b[\s!.,]*$", re.I)
_COMPLETE_SENTENCE = re.compile(r"[.!?][\"')\]]?$")


def _drop_trailing_fragment(sentences: list[str]) -> list[str]:
    """Remove a trailing piece that is not a full sentence (e.g. the next section's heading)."""
    while len(sentences) > 1 and not _COMPLETE_SENTENCE.search(sentences[-1]):
        sentences = sentences[:-1]
    return sentences


def answer_from_document(question: str, hits: list[SearchHit], embedder: HashedTfidfEmbedder) -> str:
    if _GREETING.match(question):
        return GREETING_MESSAGE
    if not hits or hits[0].score < MIN_CONFIDENT_SCORE:
        return NOT_FOUND_MESSAGE

    best = hits[0].chunk
    marker = ANSWER_MARKER.search(best.text)
    if marker:
        # FAQ entry: the answer is everything after "A:".
        sentences = _drop_trailing_fragment(split_sentences(best.text[marker.end():]))
    else:
        # Plain text: start at the sentence closest to the question, keep a little context.
        sentences = split_sentences(best.text)
        scores = embedder.transform(sentences) @ embedder.transform([question])[0]
        start = int(scores.argmax())
        sentences = _drop_trailing_fragment(sentences[start : start + 3])

    answer = " ".join(sentences).strip()
    return answer or NOT_FOUND_MESSAGE


def stream_words(text: str, delay: float = 0.03) -> Iterator[str]:
    """Yield text word by word so Demo Mode types out like a live AI answer."""
    for word in re.findall(r"\S+\s*", text):
        yield word
        time.sleep(delay)
