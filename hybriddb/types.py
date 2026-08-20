"""Public HybridDB types and constants."""

from dataclasses import dataclass
from enum import Enum

TEXT = "TEXT"
LONGTEXT = "LONGTEXT"
INTEGER = "INTEGER"
REAL = "REAL"
BOOLEAN = "BOOLEAN"
JSON = "JSON"


class SearchMode(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


KEYWORD = SearchMode.KEYWORD
SEMANTIC = SearchMode.SEMANTIC
HYBRID = SearchMode.HYBRID


@dataclass(frozen=True)
class Column:
    """Typed schema column helper for create_table()."""

    type: str
    constraints: str = ""

    def __str__(self) -> str:
        return f"{self.type} {self.constraints}".strip()


class EmbeddingModelError(Exception):
    pass
