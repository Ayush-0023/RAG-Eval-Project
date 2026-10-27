
import os
import re
import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR = "data"
DB_DIR = "chroma_store"
COLLECTION_NAME = "campusx_transcripts"


# 1. LOAD — read transcripts and remove VTT timestamps
def load_transcripts():
    docs = []

    for path in sorted(glob.glob(f"{DATA_DIR}/*.vtt")):
        lines = []

        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if (
                    not line
                    or line == "WEBVTT"
                    or "-->" in line
                    or line.isdigit()
                ):
                    continue

                lines.append(line)

        text = " ".join(lines)

        match = re.search(r"Session[\s_-]*(\d+)", path, re.I)
        session = match.group(1) if match else os.path.basename(path)

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"session": str(session)},
                )
            )

    if not docs:
        raise FileNotFoundError(
            f"No readable .vtt transcripts found in {DATA_DIR}/"
        )

    return docs


# 2. BUILD — embed chunks and persist the vector store
def load_store():
    embeddings = HuggingFaceEmbeddings(
       model_name = "BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Reuse an existing database if one exists.
    if os.path.exists(DB_DIR):
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
    )


# 3. RETRIEVE
def build_retriever():
    return load_store().as_retriever(
        search_kwargs={"k": 5}
    )


# 4. TEST — python src/retriever.py
if __name__ == "__main__":
    retriever = build_retriever()

    results = retriever.invoke("What is regression testing?")

    for result in results:
        print(
            f"[Session {result.metadata['session']}] "
            f"{result.page_content[:150]}...\n"
        )
