from __future__ import annotations

from typing import List, Dict, Any

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """DuckDuckGo search (no API key). Returns list of {title, href, body}."""
    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
    return results


def fetch_page_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)

