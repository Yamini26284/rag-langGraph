"""Run once: removes the leftover smoke-test vector from the Pinecone
index so it doesn't pollute real retrieval."""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "rag-langgraph"))
index.delete(ids=["smoke-test-1"])
print("Deleted smoke-test-1 from index.")