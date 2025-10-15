from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..storage import Document


@dataclass
class ConnectorConfig:
    type: str
    name: str
    params: Dict[str, Any]


class BaseConnector:
    def __init__(self, config: ConnectorConfig):
        self.config = config

    def fetch(self) -> List[Document]:  # pragma: no cover - abstract
        raise NotImplementedError

