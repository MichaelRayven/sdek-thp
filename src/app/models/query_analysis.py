from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class QueryAnalysis:
    country: str | None
    is_location_dependent: bool
