# AI Customer Support Agent (RAG Chatbot)

A chatbot that answers customer questions using **only your company's own document**.
Upload a PDF (FAQ, product catalog, policy manual), and customers can ask questions in
plain language and get instant answers with the source passage shown.

**Live demo:** https://pdf-support-chatbot-demo.streamlit.app/ (if it has been idle, Streamlit
takes a few seconds to wake it up). Click **Use Sample Company FAQ PDF** and ask a question.

- **Grounded answers** - replies come from the uploaded document, not from the AI's imagination.
- **Works with Anthropic Claude, OpenAI or Google Gemini** - choose with one line in `.env`.
- **Demo Mode** - runs with no API key at all, for testing and demos.
- **Private indexing** - the document is indexed on your own machine; nothing is uploaded to index it.

## Quick start (Windows)

Requires [Python 3.10 or newer](https://www.python.org/downloads/).

**Easiest:** double-click `run.bat`. The first run installs everything, then the app opens in your browser.

**Or run the commands yourself** from this folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

The app opens at http://localhost:8501. Press `Ctrl+C` in the terminal to stop it.

## Try it in 30 seconds

1. In the sidebar, click **Use Sample Company FAQ PDF** (a FAQ for a fictional store, BrightNest Home).
2. Click one of the sample questions, or type your own, for example:
   - How long does shipping take?
   - What is your return policy?
   - Do you deliver to Canada?
3. Open **Sources from the document** under an answer to see the passage it came from.
4. Ask something the document does not cover (for example "Do you sell car tires?") and the bot says it does not know instead of guessing.

To use your own document, upload any text-based PDF in the sidebar.

## Turn on AI answers (optional)

Without an API key the app runs in **Demo Mode**: it finds the best matching passage and quotes the
answer from it. With an API key, an AI model reads the matching passages and writes a natural reply,
handles follow-up questions and copes with wording that differs from the document.

1. Copy `.env.example` to `.env`.
2. Paste **one** key:
   - Anthropic: `ANTHROPIC_API_KEY=...` (get one at https://console.anthropic.com)
   - OpenAI: `OPENAI_API_KEY=...` (get one at https://platform.openai.com)
   - Google Gemini: `GEMINI_API_KEY=...` (https://aistudio.google.com/apikey)
3. Restart the app. The sidebar shows **AI Mode** and the model in use. The Demo Mode switch stays
   available so you can compare both.

Never share your `.env` file or commit it to Git; `.gitignore` already excludes it.

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Which provider to use when more than one key is set: `anthropic`, `openai` or `gemini` |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | `claude-sonnet-5-5` and `claude-haiku-5-5` cost less |
| `OPENAI_MODEL` | `gpt-4o-mini` | Set to a chat model available on your account |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Any Gemini model your key can use |
| `TOP_K` | `4` | Document sections sent to the AI per question |
| `PUBLIC_DEMO` | `false` | Set to `true` when hosting publicly: shows a privacy notice and caps questions per visit |
| `SESSION_QUESTION_LIMIT` | `15` when public | Questions allowed per visit; reloading the page resets it |

Gemini offers a free tier. On it, Google may use the text you send to improve its products, so use the free tier for testing with sample data and a paid key for real client data.

## How it works

```
PDF -> text per page -> chunks -> vectors -> vector index
                                                  |
question -> vector -> top matching chunks -> AI model (or Demo Mode) -> answer + sources
```

| File | Role |
|---|---|
| `app.py` | Streamlit interface: upload, chat, sources |
| `rag/pdf_loader.py` | Reads the PDF and splits it; FAQ documents get one chunk per question |
| `rag/embeddings.py` | Local hashed TF-IDF embeddings (NumPy only, no downloads, no cost) |
| `rag/vector_store.py` | Similarity search; uses FAISS if installed, NumPy otherwise |
| `rag/llm.py` | Anthropic / OpenAI / Gemini streaming, with the "answer only from the document" prompt |
| `rag/demo.py` | Demo Mode answers |
| `rag/config.py` | Settings from `.env` |
| `scripts/generate_sample_pdf.py` | Recreates `sample_company_faq.pdf` |
| `tests/smoke_test.py` | End-to-end check, no API key needed |

Run the check with:

```powershell
.venv\Scripts\python.exe tests\smoke_test.py
```

## Notes for developers

- **Search quality.** The built-in embeddings match on weighted keywords plus a small synonym table
  (`_CANONICAL_TERMS` in `rag/embeddings.py`), so a question worded very differently from the document
  can miss, and Demo Mode can occasionally quote a related but wrong passage. For production use,
  swap `HashedTfidfEmbedder` for a semantic embedding model; anything with `fit_transform` and
  `transform` methods drops in.
- **FAISS is optional.** `pip install faiss-cpu` and the store uses it automatically. It needs a
  Python version FAISS publishes wheels for; without it the NumPy index gives identical results.
- **Refusal fallback.** For current Claude models the Anthropic call opts in to the server-side
  fallback (`fallbacks="default"`), so a request declined by a safety classifier is retried on
  another model instead of failing. Remove `betas` and `fallbacks` in `rag/llm.py` to turn it off.
- **Scanned PDFs** contain images, not text, and are rejected with a clear message. Add OCR to support them.
- **One document, in memory.** The index is rebuilt per session. For many documents or many users,
  move to a persistent vector database.
