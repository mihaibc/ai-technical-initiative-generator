from __future__ import annotations

from typing import List, Optional, Dict, Any

from .embeddings import embed_texts, top_k_similar
from .models import LLMClient, LLMConfig
from .prompts import build_generation_prompt, DEFAULT_CATEGORIES, SYSTEM_PROMPT
from .storage import load_index
from .web import web_search, fetch_page_text


def _collect_internal_context(query: str, k: int = 6) -> List[str]:
    vecs, meta = load_index()
    if vecs is None or len(meta) == 0:
        return []
    qv = embed_texts([query])[0]
    idxs = top_k_similar(qv, vecs, k=k)
    return [meta[i]["text"][:1200] for i in idxs]


def _collect_web_context(query: str, k: int = 3) -> List[str]:
    hits = web_search(query, max_results=k)
    snippets: List[str] = []
    for h in hits:
        url = h.get("href")
        try:
            text = fetch_page_text(url)
            if text:
                snippets.append(text[:1500])
        except Exception:
            continue
    return snippets


class InitiativeAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(LLMConfig())

    def generate(
        self,
        objective: str,
        categories: Optional[List[str]] = None,
        use_internal: bool = True,
        use_web: bool = False,
        constraints: Optional[List[str]] = None,
        num_per_category: int = 3,
    ) -> Dict[str, Any]:
        cats = categories or DEFAULT_CATEGORIES
        context_snippets: List[str] = []
        if use_internal:
            context_snippets.extend(_collect_internal_context(objective, k=6))
        if use_web:
            context_snippets.extend(_collect_web_context(objective, k=3))

        prompt = f"{SYSTEM_PROMPT}\n\n" + build_generation_prompt(
            objective=objective,
            categories=cats,
            context_snippets=context_snippets,
            constraints=constraints,
            num_per_category=num_per_category,
        )
        output = self.llm.generate(prompt)
        return {
            "prompt": prompt,
            "output_markdown": output,
            "used_internal": use_internal,
            "used_web": use_web,
            "context_count": len(context_snippets),
        }

