from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from .base import BaseConnector, ConnectorConfig
from ..storage import Document


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


class HTTPConnector(BaseConnector):
    """Fetches text from a URL with optional auth.

    params expected:
      - url: str
      - auth_type: str in {none, basic, bearer}
      - username/password for basic
      - token for bearer
      - headers: dict (optional)
    """

    def fetch(self) -> List[Document]:
        p: Dict[str, Any] = self.config.params
        url = p["url"]
        auth_type = (p.get("auth_type") or "none").lower()
        headers: Dict[str, str] = p.get("headers", {})
        auth = None
        if auth_type == "basic":
            auth = (p.get("username", ""), p.get("password", ""))
        elif auth_type == "bearer":
            token = p.get("token", "")
            headers = {**headers, "Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, auth=auth, timeout=30)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "html" in content_type:
            text = _extract_text_from_html(r.text)
        else:
            text = r.text
        return [Document(text=text, source=url, metadata={"content_type": content_type})]

