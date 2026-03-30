from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class KnowledgeRecord:
    id: str
    content: str
    metadata: dict[str, Any]