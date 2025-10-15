from typing import List, Optional


DEFAULT_CATEGORIES = [
    "Data & Analytics",
    "Product Features",
    "Platform Health",
    "Architecture & Scalability",
    "Developer Experience",
    "Security & Compliance",
    "Process & Culture",
    "Experimentation & Growth",
]


SYSTEM_PROMPT = (
    "You are a senior engineering leader and AI strategy partner. "
    "You translate business goals into concrete, actionable technical initiatives with clear impact metrics. "
    "Generate pragmatic, high‑leverage initiatives, not generic platitudes. "
    "Favor realistic scoping, ROI framing, and measurable outcomes."
)


def build_generation_prompt(
    objective: str,
    categories: Optional[List[str]] = None,
    context_snippets: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    num_per_category: int = 3,
) -> str:
    cats = categories or DEFAULT_CATEGORIES
    ctx = "\n\n".join([f"- {c}" for c in (context_snippets or [])])
    cons = "\n".join([f"- {c}" for c in (constraints or [])])

    prompt = f"""
Objective:
{objective}

Additional Context (optional):
{ctx if ctx else '(none)'}

Constraints (optional):
{cons if cons else '(none)'}

Task:
Translate the objective into a concise, categorized initiative plan. For each category below, propose up to {num_per_category} high‑impact initiatives. Each item must include:
- Initiative name (bold, few words)
- One‑sentence description
- Why now (business value / risk)
- Expected impact (with a measurable leading metric)
- Effort level (S/M/L)

Categories:
{chr(10).join(['- ' + c for c in cats])}

Output strictly in Markdown using this format:

### {{Category}}
- **{{Initiative Name}}** — {{description}}
  - Why now: {{value}}
  - Impact: {{metric}}
  - Effort: {{S|M|L}}

Only include categories that have at least one meaningful initiative. Avoid filler. Be specific and pragmatic.
""".strip()
    return prompt

