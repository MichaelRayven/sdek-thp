from pathlib import Path
from app.models.document import Document


class RetrieverService:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.documents = self._load_documents()

    def _load_documents(self) -> list[Document]:
        files = {
            ("general_info.txt", None),
            ("deadlines.txt", None),
            ("benefits.txt", None),
            ("germany_rules.txt", "germany"),
            ("france_rules.txt", "france"),
        }

        documents: list[Document] = []

        for filename, country in files:
            path = self.data_dir / filename
            documents.append(
                Document(
                    source=filename,
                    content=path.read_text(encoding="utf-8"),
                    country=country,
                )
            )

        return documents

    def retrieve(self, country: str | None) -> list[Document]:
        docs: list[Document] = []

        for doc in self.documents:
            if doc.country is None:
                docs.append(doc)

        if country is not None:
            docs.extend(doc for doc in self.documents if doc.country == country)

        return docs
