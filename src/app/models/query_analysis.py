from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class QueryAnalysis:
    country: str | None
    needs_clarification: bool
