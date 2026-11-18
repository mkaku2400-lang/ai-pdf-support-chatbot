"""Generate grounded answers with Anthropic Claude, OpenAI or Google Gemini."""

from __future__ import annotations

from collections.abc import Iterator

from .config import Settings
from .vector_store import SearchHit

SYSTEM_PROMPT = """You are a customer support assistant for a business. Customers ask you \
questions, and with each question you receive numbered excerpts from the company's own document.

Answer using only the information in those excerpts. The company is legally and commercially \
responsible for what you tell customers, so never add policies, prices, dates or promises that \
the excerpts do not state, even if they seem likely.

If the excerpts do not contain the answer, say you don't have that information and suggest \
contacting the support team; share the support contact details if the excerpts include them.

Write the way a friendly, professional support agent would: short, clear and direct, in the \
customer's language. Do not mention "excerpts" or "the document" mechanics; just answer. \
Begin each reply with the answer itself. Greet the customer only when their message is just a greeting; in the middle of a conversation a fresh \"Hello\" reads as if you forgot them."""

# Models that accept the server-side refusal fallback (see README, "Notes for developers").
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}
)

MAX_HISTORY_MESSAGES = 6
MAX_OUTPUT_TOKENS = 16000


class LLMError(Exception):
    """An AI provider failure, with a message safe to show the user."""


def _build_messages(question: str, hits: list[SearchHit], history: list[dict]) -> list[dict]:
    if hits:
        excerpts = "\n\n".join(
            f"[{i}] (page {hit.chunk.page})\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
        )
    else:
        excerpts = "(No relevant excerpts were found for this question.)"
    user_message = f"Company document excerpts:\n\n{excerpts}\n\nCustomer question: {question}"

    recent = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-MAX_HISTORY_MESSAGES:]
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    # The API requires the conversation to start with a user message.
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    return [*recent, {"role": "user", "content": user_message}]


def _stream_anthropic(settings: Settings, messages: list[dict]) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    request = dict(
        model=settings.anthropic_model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        # Support answers are short lookups; low effort keeps replies fast and cheap.
        output_config={"effort": "low"},
    )
    if settings.anthropic_model in _FALLBACK_MODELS:
        stream_manager = client.beta.messages.stream(**request, betas=[_FALLBACK_BETA], fallbacks="default")
    else:
        stream_manager = client.messages.stream(**request)

    try:
        with stream_manager as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in your .env file.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMError("This Anthropic API key is not allowed to use the selected model.") from exc
    except anthropic.NotFoundError as exc:
        raise LLMError(f"Model '{settings.anthropic_model}' was not found. Check ANTHROPIC_MODEL in .env.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("The AI service is busy (rate limit reached). Please wait a moment and try again.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"The AI service returned an error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the AI service. Check your internet connection.") from exc

    if final.stop_reason == "refusal":
        yield "I'm not able to help with that request. Please contact our support team directly."
    elif final.stop_reason == "max_tokens":
        yield "\n\n*(The answer was cut short because it reached the length limit.)*"


def _stream_openai(settings: Settings, messages: list[dict]) -> Iterator[str]:
    import openai

    client = openai.OpenAI(api_key=settings.openai_api_key)
    try:
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
    except openai.AuthenticationError as exc:
        raise LLMError("The OpenAI API key was rejected. Check OPENAI_API_KEY in your .env file.") from exc
    except openai.NotFoundError as exc:
        raise LLMError(f"Model '{settings.openai_model}' was not found. Check OPENAI_MODEL in .env.") from exc
    except openai.RateLimitError as exc:
        raise LLMError("The AI service is busy or out of credit (rate limit). Please try again later.") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"The AI service returned an error ({exc.status_code}).") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("Could not reach the AI service. Check your internet connection.") from exc


def _stream_gemini(settings: Settings, messages: list[dict]) -> Iterator[str]:
    from google import genai
    from google.genai import errors, types

    # Gemini returns 503 when a model is busy, which is common on the free tier, so retry a few times.
    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0)),
    )
    # Gemini calls the assistant's turns "model".
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    try:
        stream = client.models.generate_content_stream(
            model=settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except errors.APIError as exc:
        message = str(getattr(exc, "message", "") or exc)
        if exc.code in (401, 403) or "API key" in message:
            raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY in your .env file.") from exc
        if exc.code == 404:
            raise LLMError(f"Model '{settings.gemini_model}' was not found. Check GEMINI_MODEL in .env.") from exc
        if exc.code == 429:
            raise LLMError("The Gemini usage limit was reached (the free tier allows a few requests per minute). Wait a minute and try again.") from exc
        if exc.code in (500, 502, 503, 504):
            raise LLMError("Gemini is overloaded right now. Wait a minute and try again.") from exc
        raise LLMError(f"Gemini returned an error ({exc.code}): {message}") from exc
    except Exception as exc:  # noqa: BLE001 - the SDK does not wrap network failures in its own error type
        raise LLMError(f"Could not reach Gemini. Check your internet connection. ({type(exc).__name__})") from exc


def stream_answer(settings: Settings, question: str, hits: list[SearchHit], history: list[dict]) -> Iterator[str]:
    """Stream an answer grounded in the retrieved excerpts. Raises LLMError on failure."""
    messages = _build_messages(question, hits, history)
    if settings.provider == "anthropic":
        yield from _stream_anthropic(settings, messages)
    elif settings.provider == "openai":
        yield from _stream_openai(settings, messages)
    elif settings.provider == "gemini":
        yield from _stream_gemini(settings, messages)
    else:
        raise LLMError("No AI provider is configured. Add an API key to .env or switch on Demo Mode.")
