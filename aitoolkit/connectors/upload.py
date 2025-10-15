from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import pandas as pd

from .base import BaseConnector, ConnectorConfig
from ..storage import Document


def _text_from_csv_bytes(b: bytes, max_rows: int = 500) -> str:
    df = pd.read_csv(io.BytesIO(b))
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_csv(index=False)


def _text_from_json_bytes(b: bytes) -> str:
    try:
        data = json.loads(b.decode("utf-8"))
        return json.dumps(data, indent=2)
    except Exception:
        return b.decode("utf-8", errors="ignore")


class UploadConnector(BaseConnector):
    """Handles CSV/JSON/TXT uploads provided as bytes in params.

    params expected:
      - filename: str
      - mime: str
      - content: bytes
    """

    def fetch(self) -> List[Document]:
        p: Dict[str, Any] = self.config.params
        filename = p.get("filename", "upload")
        mime = p.get("mime", "text/plain")
        content: bytes = p.get("content", b"")
        text = ""
        if mime in ("text/csv", "application/csv") or filename.endswith(".csv"):
            text = _text_from_csv_bytes(content)
        elif mime in ("application/json",) or filename.endswith(".json"):
            text = _text_from_json_bytes(content)
        else:
            text = content.decode("utf-8", errors="ignore")
        return [
            Document(
                text=text,
                source=f"upload://{filename}",
                metadata={"filename": filename, "mime": mime},
            )
        ]

