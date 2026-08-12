"""
Runs every question in eval/self_test_questions.json against the live
API (must be running: uvicorn app.main:app --reload) and writes a
pass/fail report to eval/results.md.

Checks two things per question:
  1. Does the answer contain the expected key phrase(s)? (word-level
     match, tolerant of minor phrasing like "and" between list items)
  2. Do the citations include the expected source file(s)?
     (for refusal/adversarial cases, expects citations to be empty)

A short delay between requests avoids tripping Gemini's free-tier
rate limit when running all 15 back-to-back.

Run: python eval/run_eval.py
"""
import json
import time
from pathlib import Path

import requests

DELAY_SECONDS = 4  # spacing between requests to stay under free-tier rate limits

QUESTIONS_PATH = Path(__file__).parent / "self_test_questions.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
API_URL = "http://127.0.0.1:8000/ask"


def check(question: dict, response: dict) -> tuple[bool, str]:
    answer = response.get("answer", "").lower()
    citations = response.get("citations", [])
    cited_files = {c["source_file"] for c in citations}
    expected_files = set(question["expected_source_files"])
    expected_substr = question["expected_answer_contains"].lower()

    # tolerant match: every key word/phrase (split on commas) must appear,
    # ignoring minor connectors like "and" — avoids false negatives from
    # natural phrasing differences while still catching real mismatches
    key_parts = [p.strip() for p in expected_substr.split(",") if p.strip()]
    answer_ok = all(part in answer for part in key_parts)

    if question["category"] in ("adversarial", "out_of_corpus"):
        citations_ok = len(citations) == 0
    else:
        citations_ok = expected_files.issubset(cited_files)

    passed = answer_ok and citations_ok
    reason = []
    if not answer_ok:
        reason.append(f"expected answer to contain '{question['expected_answer_contains']}'")
    if not citations_ok:
        reason.append(f"expected citations {expected_files or '(none)'}, got {cited_files or '(none)'}")
    return passed, "; ".join(reason) if reason else "ok"


def fetch_answer(question: str) -> dict | None:
    """POST to /ask with one retry on timeout/connection error."""
    for attempt in range(2):
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            data = resp.json()
            if resp.status_code != 200:
                return {"_error": f"HTTP {resp.status_code}: {data.get('detail') or data.get('error')}"}
            return data
        except Exception as e:
            if attempt == 0:
                time.sleep(DELAY_SECONDS)
                continue
            return {"_error": f"request error: {e}"}


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    lines = ["# Eval Results\n"]
    passed_count = 0

    for q in questions:
        data = fetch_answer(q["question"])

        if data is None or "_error" in data:
            err = data["_error"] if data else "unknown error"
            lines.append(f"## Q{q['id']}: {q['question']}\n- **FAIL** ({err})\n")
            time.sleep(DELAY_SECONDS)
            continue

        passed, reason = check(q, data)
        passed_count += int(passed)
        status = "PASS" if passed else "FAIL"

        lines.append(f"## Q{q['id']}: {q['question']}")
        lines.append(f"- Category: {q['category']}")
        lines.append(f"- **{status}** — {reason}")
        lines.append(f"- Answer: {data.get('answer', '')}")
        lines.append(f"- Citations: {[c['source_file'] for c in data.get('citations', [])]}")
        lines.append(f"- Grade reason: {data.get('trace', {}).get('grade_reason', '')}")
        lines.append("")
        time.sleep(DELAY_SECONDS)

    lines.insert(1, f"**{passed_count}/{len(questions)} passed**\n")
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"{passed_count}/{len(questions)} passed. Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()