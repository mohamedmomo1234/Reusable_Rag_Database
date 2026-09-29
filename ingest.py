from pathlib import Path
import shutil

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
)
from embeddings import get_embeddings


def load_documents():
    documents = []
    docs_path = Path(DOCS_DIR)

    if not docs_path.exists():
        return documents

    for file in sorted(docs_path.rglob("*")):
        if not file.is_file():
            continue

        extension = file.suffix.lower()

        try:
            if extension == ".pdf":
                docs = PyPDFLoader(str(file)).load()
            elif extension == ".txt":
                docs = TextLoader(str(file), encoding="utf-8").load()
            elif extension == ".docx":
                docs = Docx2txtLoader(str(file)).load()
            else:
                continue

            for doc in docs:
                doc.metadata["source_file"] = file.name
                doc.metadata["file_type"] = extension
                doc.metadata["document_type"] = "reference"
                doc.metadata["source_path"] = str(file)

            documents.extend(docs)

        except Exception as exc:
            print(f"Could not load {file.name}: {exc}")

    return documents


def create_vector_database(reset=True):
    documents = load_documents()

    if not documents:
        raise ValueError(f"No supported documents found in {DOCS_DIR}")

    if reset:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = int(chunk.metadata["page"]) + 1

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Embedding model: {get_embeddings().__class__.__name__}")
    print("Reusable RAG knowledge base created successfully.")


if __name__ == "__main__":
    create_vector_database(reset=True)
