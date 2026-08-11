"""
Phase 0 smoke test: proves Gemini embeddings + Pinecone upsert/query work
before any real ingestion or graph logic is built.
"""
import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-langgraph")
DIM = int(os.environ.get("EMBEDDING_DIM", 768))
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 1. Create index if it doesn't exist
if not pc.has_index(INDEX_NAME):
    print(f"Creating index '{INDEX_NAME}' (dim={DIM})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)

# 2. Embed a test string
test_text = "The Smith case was heard in the Northern District Court."
result = client.models.embed_content(
    model=os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001"),
    contents=test_text,
    config={"output_dimensionality": DIM},
)
vector = result.embeddings[0].values
print(f"Embedded test string -> vector length {len(vector)}")

# 3. Upsert it
index.upsert(vectors=[{
    "id": "smoke-test-1",
    "values": vector,
    "metadata": {"chunk_id": "smoke-test-1", "source_file": "smoke_test.py"},
}])
print("Upserted test vector.")

# 4. Query it back
query_result = index.query(vector=vector, top_k=1, include_metadata=True)
print("Query result:", query_result)

assert query_result["matches"][0]["id"] == "smoke-test-1"
print("\n✅ Smoke test passed: Gemini + Pinecone chain works end-to-end.")