"""Build a local FAISS index from JSONL and PDF files in data/."""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDEX = ROOT / "faiss_index"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    documents = []
    for path in sorted(DATA.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected a JSON object")
                    parts = [f"Question: {record.get('instruction', '')}"]
                    if record.get("input"):
                        parts.append(f"Additional input: {record['input']}")
                    parts.append(f"Answer: {record.get('output', '')}")
                    if not record.get("instruction") and not record.get("output"):
                        continue
                    documents.append(Document(page_content="\n".join(parts), metadata={"source": path.name, "record": number}))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path.name}, line {number}: {exc}") from exc
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    for path in sorted(DATA.glob("*.pdf")):
        reader = PdfReader(str(path))
        for number, page in enumerate(reader.pages, 1):
            content = page.extract_text() or ""
            if content.strip():
                documents.extend(splitter.split_documents([Document(page_content=content, metadata={"source": path.name, "page": number})]))
    return documents


def main():
    documents = load_documents()
    if not documents:
        raise SystemExit("No readable knowledge found. Add data/train.jsonl or a text-based PDF to data/.")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL)
    FAISS.from_documents(documents, embeddings).save_local(str(INDEX))
    print(f"Indexed {len(documents)} searchable passages in {INDEX}")


if __name__ == "__main__":
    main()
