from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    source: str
    content: str
    country: str | None = None
