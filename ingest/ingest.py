"""
Ingestion pipeline.

What this does, in order:
1. Reads every file in corpus/
2. Splits each file into chunks along markdown headings (structure-aware,
   not blind fixed-size windows) — keeps each chunk semantically whole.
3. Builds a deterministic point ID per chunk (hash of file name + chunk
   index), so re-running ingest overwrites the same vectors instead of
   creating duplicates. Answers the brief's "what happens on double
   ingest?" question by construction.
4. Embeds each chunk with Gemini and upserts to Pinecone with metadata:
   chunk_id, source_file, section heading, and the raw text (so the
   generation node can quote it back without a second lookup).

"""

import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-langgraph")
DIM = int(os.environ.get("EMBEDDING_DIM", 768))
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


def chunk_markdown(text: str, source_file: str) -> list[dict]:
    """
    Split markdown text on heading boundaries (## level, falling back to
    # level if a file has no ##). Each chunk keeps its heading as a
    prefix so embeddings capture the section context, not just the
    raw body text.
    """
    lines = text.splitlines()
    chunks = []
    current_heading = "Introduction"
    current_body = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            chunks.append({"heading": current_heading, "text": f"{current_heading}\n{body}"})

    for line in lines:
        if re.match(r"^##\s+", line):
            flush()
            current_heading = line.lstrip("#").strip()
            current_body = []
        elif re.match(r"^#\s+", line):
            # top-level title — treat as heading context, don't split further
            current_heading = line.lstrip("#").strip()
        else:
            current_body.append(line)

    flush()
    return chunks


def make_chunk_id(source_file: str, index: int) -> str:
    """Deterministic ID: same file + same chunk index always -> same ID.
    Re-running ingest upserts over the same vectors instead of duplicating."""
    raw = f"{source_file}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embed(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"output_dimensionality": DIM},
    )
    return result.embeddings[0].values


def ensure_index():
    if not pc.has_index(INDEX_NAME):
        print(f"Creating index '{INDEX_NAME}' (dim={DIM})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    return pc.Index(INDEX_NAME)


def main():
    index = ensure_index()

    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {CORPUS_DIR}. Did you extract the corpus there?")
        return

    total_chunks = 0
    for filepath in files:
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, filepath.name)

        vectors = []
        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(filepath.name, i)
            vector = embed(chunk["text"])
            vectors.append({
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": filepath.name,
                    "heading": chunk["heading"],
                    "text": chunk["text"],
                },
            })

        if vectors:
            index.upsert(vectors=vectors)
            total_chunks += len(vectors)
            print(f"{filepath.name}: {len(vectors)} chunks upserted")

    print(f"\n✅ Ingest complete. {total_chunks} chunks across {len(files)} files.")


if __name__ == "__main__":
    main()