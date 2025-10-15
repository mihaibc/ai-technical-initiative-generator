# AI-Powered Technical Initiative Generator

Turn a business objective into a pragmatic, categorized initiative plan for engineering leaders. Built for Hugging Face Spaces with a professional Gradio UI, modular agent, web research, and pluggable data sources.

Concept: The Engineering Leader's AI Toolkit — Pillar 2 (Strategy)

## Features
- Initiative generation from a single objective into clear categories (Data & Analytics, Product, Platform Health, etc.)
- Optional grounding using your own data sources (CSV/JSON uploads, URLs with Basic/Bearer auth)
- Optional web research via DuckDuckGo search and content extraction
- Lightweight, configurable LLM stack: Hugging Face Inference API, local Transformers, or external providers
- Clean Gradio interface with debug prompt and details panel

## Quick Start (Hugging Face Spaces)
1. Create a new Space (Gradio) and push the repo contents.
2. In the Space Settings, set the following (optional) secrets:
   - `HUGGINGFACEHUB_API_TOKEN` — recommended for Inference API
   - `HF_INFERENCE_MODEL` — default: `Qwen/Qwen2.5-3B-Instruct`
   - `EMBEDDING_MODEL` — default: `sentence-transformers/all-MiniLM-L6-v2`
   - `LLM_PROVIDER` — one of: `hf_inference` (default), `local`, `openai`, `together`, `groq`
   - Optionally: `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `GROQ_API_KEY`
3. Hardware: CPU is OK. Enable Internet if you want web research and URL connectors to fetch content.
4. Space auto-detects `app.py` and launches the Gradio UI.

## Local Run
```
pip install -r requirements.txt
python app.py
```

## Usage
1. Generate Initiatives
   - Enter a business objective (e.g., “Reduce customer churn by 10%”).
   - Choose categories, set initiatives per category.
   - Toggle “Use Data Sources” and/or “Use Web Research”.
   - Click Generate to get a Markdown plan with Why/Impact/Effort for each item.

2. Data Sources
   - Upload: CSV/JSON/TXT are embedded and added to a lightweight vector store.
   - URL: Fetches public pages or API responses; supports `none`, `basic`, and `bearer` auth.
   - Multiple sources are supported; added content improves grounding and specificity.

## Model Providers
- Hugging Face Inference API (default): set `HUGGINGFACEHUB_API_TOKEN` and `HF_INFERENCE_MODEL`.
- Local Transformers: set `provider=local` in code or env and ensure hardware is sufficient.
- External APIs: OpenAI, Together, Groq are supported if their API keys are set and the provider is selected in code.

The default model balances quality and weight: `Qwen/Qwen2.5-3B-Instruct`.

## Architecture
- `app.py` — Space entry; launches the Gradio UI
- `aitoolkit/ui.py` — UI assembly and event wiring
- `aitoolkit/agent.py` — Initiative agent; retrieval + (optional) web research + LLM
- `aitoolkit/models.py` — Pluggable LLM client (HF Inference, local, external)
- `aitoolkit/prompts.py` — System + task prompt templates
- `aitoolkit/embeddings.py` — SentenceTransformer embeddings + cosine search
- `aitoolkit/storage.py` — Simple JSON/NumPy persistence for connections and vectors
- `aitoolkit/connectors/` — Upload and HTTP connectors (basic/bearer)
- `requirements.txt` — Dependencies

## Security Notes
- Uploaded file bytes are not stored verbatim; only text and metadata are embedded and saved. Still, do not upload sensitive data to public Spaces.
- URL connector supports Basic and Bearer token auth; secrets should be provided securely (e.g., Space Secrets or private Space).
- Runtime “connection script” execution is intentionally not supported for safety. To add custom connectors, implement a new module under `aitoolkit/connectors/` and wire it in `ui.py`.

## Roadmap
- Additional connectors (GitHub, Notion, Google Drive)
- Structured JSON output option and richer export
- Per-initiative cost/benefit scoring and timeline suggestions
- In-app provider selection and settings persistence

## License
See `LICENSE`.
