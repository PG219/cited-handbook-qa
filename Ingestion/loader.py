from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List


def load_pdf(filename: str) -> List[Document]:
    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = base_dir / "DATA_1" / "RAW" / filename

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    return docs
