from app.models.query_analysis import QueryAnalysis


COUNTRY_ALIASES = {
    "germany": [
        "germany",
        "german",
        "berlin",
        "германия",
        "германии",
        "берлин",
        "берлине",
    ],
    "france": [
        "france",
        "french",
        "paris",
        "франция",
        "франции",
        "париж",
        "париже",
    ],
}

LOCATION_SPECIFIC_TERMS = [
    "scholarship",
    "stipend",
    "tax",
    "working day",
    "workday",
    "visa",
    "стипендия",
    "стипендии",
    "налог",
    "рабочий день",
    "виза",
]


class QueryAnalyzerService:
    def analyze(self, text: str) -> QueryAnalysis:
        normalized = text.lower()

        country = self._detect_country(normalized)
        is_location_dependent = self._is_location_dependent(normalized)
        needs_clarification = is_location_dependent and country is None

        return QueryAnalysis(
            country=country,
            needs_clarification=needs_clarification,
        )

    def _detect_country(self, text: str) -> str | None:
        for country, aliases in COUNTRY_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    return country

        return None

    def _is_location_dependent(self, text: str) -> bool:
        for term in LOCATION_SPECIFIC_TERMS:
            if term in text:
                return True

        return False
