"""
LangGraph flow for rag-langgraph.

Nodes:
  retrieve  -> embed question, query Pinecone top-k chunks
  grade     -> decide if chunks are good enough. Two signals, not one:
               (1) average similarity score vs threshold
               (2) a cheap LLM check: "do these chunks actually answer this?"
               Always returns a *reason*, not just pass/fail — this is
               what lets the refuse path explain itself instead of just
               saying "no."
  generate  -> good path: write answer, cite chunk_id + source_file for
               every claim, using ONLY retrieved text.
  refuse    -> bad path after retries exhausted: say plainly the docs
               don't answer this, include the grade reason.

Branch: grade -> generate (good) OR retry-retrieve OR refuse (bad,
no retries left). This is a real conditional edge, not an if-statement
buried inside one node.

Loop guard: state["retries"] is capped at MAX_RETRIES. Once hit, the
graph is forced to refuse rather than retry indefinitely.
"""
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, END
from pinecone import Pinecone

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-langgraph")
DIM = int(os.environ.get("EMBEDDING_DIM", 768))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-2.5-flash")

TOP_K = 4
HIGH_CONFIDENCE = 0.68   # above this, trust similarity alone — skip the LLM check
LOW_CONFIDENCE = 0.50    # below this, reject immediately — skip the LLM check
# between the two: genuinely uncertain, worth spending an LLM call to check
MAX_RETRIES = 2                # loop guard
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


class GraphState(TypedDict):
    question: str
    retries: int
    chunks: list[dict]
    grade_ok: bool
    grade_reason: str
    answer: str
    citations: list[dict]


def embed(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"output_dimensionality": DIM},
    )
    return result.embeddings[0].values


def retrieve_node(state: GraphState) -> GraphState:
    vector = embed(state["question"])
    result = index.query(vector=vector, top_k=TOP_K, include_metadata=True)
    chunks = [
        {
            "chunk_id": m["metadata"].get("chunk_id", m["id"]),
            "source_file": m["metadata"].get("source_file", "unknown"),
            "heading": m["metadata"].get("heading", ""),
            "text": m["metadata"].get("text", ""),
            "score": m["score"],
        }
        for m in result["matches"]
        if m["metadata"].get("text")  # skip any vector with no real content (e.g. stray test data)
    ]
    return {**state, "chunks": chunks}


def grade_node(state: GraphState) -> GraphState:
    chunks = state["chunks"]

    if not chunks:
        return {**state, "grade_ok": False, "grade_reason": "no chunks retrieved"}

    avg_score = sum(c["score"] for c in chunks) / len(chunks)

    # decisive cases: trust similarity alone, skip the extra LLM call
    if avg_score >= HIGH_CONFIDENCE:
        return {**state, "grade_ok": True, "grade_reason": f"passed on similarity alone (avg={avg_score:.2f}, high confidence)"}
    if avg_score < LOW_CONFIDENCE:
        return {**state, "grade_ok": False, "grade_reason": f"low similarity (avg={avg_score:.2f})"}

    # borderline case: spend one LLM call to check actual relevance,
    # not just vector proximity
    context = "\n\n".join(f"[{c['source_file']}] {c['text']}" for c in chunks)
    check_prompt = (
        f"Question: {state['question']}\n\n"
        f"Retrieved passages:\n{context}\n\n"
        "Do these passages contain information that directly answers the "
        "question? Reply with exactly one word: YES or NO."
    )
    resp = client.models.generate_content(model=GENERATION_MODEL, contents=check_prompt)
    verdict = resp.text.strip().upper()

    if "YES" not in verdict:
        return {**state, "grade_ok": False, "grade_reason": f"borderline similarity (avg={avg_score:.2f}), chunks off-topic per LLM check"}

    return {**state, "grade_ok": True, "grade_reason": f"borderline similarity (avg={avg_score:.2f}), passed LLM relevance check"}


def retry_router(state: GraphState) -> str:
    if state["grade_ok"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "refuse"
    return "retry"


def bump_retry_node(state: GraphState) -> GraphState:
    return {**state, "retries": state["retries"] + 1}


def generate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(
        f"[chunk_id={c['chunk_id']} | source={c['source_file']}]\n{c['text']}"
        for c in state["chunks"]
    )
    prompt = (
        "Answer the question using ONLY the passages below. "
        "Do not use outside knowledge.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        "Respond in exactly this format:\n"
        "ANSWER: <your answer>\n"
        "USED_CHUNK_IDS: <comma-separated chunk_ids you actually relied on, "
        "only the ones whose content you used>"
    )
    resp = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    raw = resp.text.strip()

    answer_part = raw
    used_ids = {c["chunk_id"] for c in state["chunks"]}  # fallback: all, if parsing fails

    # case-insensitive split — models don't always match "USED_CHUNK_IDS"
    # casing exactly, and an exact-string match silently falls back to
    # citing everything retrieved, defeating the point of this parsing.
    match = re.split(r"USED_CHUNK_IDS\s*:", raw, maxsplit=1, flags=re.IGNORECASE)
    if len(match) == 2:
        answer_part, ids_part = match
        answer_part = re.sub(r"^\s*ANSWER\s*:", "", answer_part, flags=re.IGNORECASE).strip()
        used_ids = {i.strip() for i in ids_part.split(",") if i.strip()}

    citations = [
        {"chunk_id": c["chunk_id"], "source_file": c["source_file"], "heading": c["heading"]}
        for c in state["chunks"]
        if c["chunk_id"] in used_ids
    ]

    return {**state, "answer": answer_part, "citations": citations}


def refuse_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": (
            "I can't find this in the provided documents "
            f"(reason: {state['grade_reason']})."
        ),
        "citations": [],
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        retry_router,
        {"generate": "generate", "retry": "bump_retry", "refuse": "refuse"},
    )

    graph.add_edge("bump_retry", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def ask(question: str) -> dict:
    app_graph = build_graph()
    result = app_graph.invoke({
        "question": question,
        "retries": 0,
        "chunks": [],
        "grade_ok": False,
        "grade_reason": "",
        "answer": "",
        "citations": [],
    })
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "trace": {
            "retries": result["retries"],
            "grade_reason": result["grade_reason"],
            "retrieved_chunks": [
                {"chunk_id": c["chunk_id"], "source_file": c["source_file"], "score": c["score"]}
                for c in result["chunks"]
            ],
        },
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the notice period in the employment agreement?"
    out = ask(q)
    print("ANSWER:", out["answer"])
    print("CITATIONS:", out["citations"])
    print("TRACE:", out["trace"])