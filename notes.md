# Decisions log

Personal notes on why things were built the way they were, what broke along the way, and what stood out. Not required reading — just here for anyone curious, and mostly for my own memory.

## Setup decisions

**Embedding size: 768, not the default 3072.**
Gemini's embedding model defaults to 3072 dimensions but supports shrinking it. Went with 768 — faster search, less storage, and for a corpus this small (6 files, 15 chunks) the accuracy difference is negligible. Every vector in Pinecone has to match the index's fixed dimension exactly, so this had to be decided before the first ingest.

**Pinecone region: aws / us-east-1.**
Just the standard free-tier default. No real reason to deviate for a project this size — would only matter if this were latency-sensitive or had a compliance requirement.

**Structure-aware chunking instead of fixed-size windows.**
Every document in the corpus already had clean markdown headings (`##`). Splitting on those headings instead of blindly cutting every N characters means each chunk is a complete thought — a clause, a section — instead of a random slice that might cut off mid-sentence. Confirmed this was the right call once I saw two files (04 and 05) don't use `##` at all, just bold text — they correctly stayed as one chunk each rather than being force-split.

**Deterministic chunk IDs instead of random ones.**
Each chunk's ID is a hash of `filename + chunk index`, not a random UUID. This means running ingest twice produces the exact same IDs, so Pinecone treats the second run as an _update_, not a duplicate insert. Verified this by running ingest twice in a row and confirming the vector count stayed at 15, not 30.

**Confidence-aware grading with two signals, not one.**
Originally the "is this good enough" check always ran a second LLM call to verify relevance, on top of the similarity score. Changed this after hitting Gemini's free-tier daily limit — now it only spends that extra LLM call when the similarity score is genuinely borderline. If the score is clearly high or clearly low, it decides immediately without the extra call. This turned out to be a better design anyway, not just a workaround — spending extra reasoning only when there's real uncertainty is a more honest way to build a confidence check.

**Citations reflect what the model actually used, not everything retrieved.**
Early version cited every retrieved chunk regardless of whether the answer actually drew from it — caught this when a question about employment notice periods cited an unrelated property lease chunk that happened to clear the similarity bar. Fixed by having the model report which chunk IDs it actually used, and filtering citations to just those.

## What broke, and what fixed it

**Leftover test data polluting real search results.**
The very first smoke-test vector (used to confirm Gemini + Pinecone were wired up correctly) never got deleted from the index. It only had partial metadata (no `heading`, no `text`), and for a completely unrelated question like "what is the capital of France," it could rank in the top matches by pure accident — since every real chunk scored low too. This caused a crash (`KeyError: 'heading'`). Fixed two ways: deleted the stray vector, and made the retrieval code read metadata defensively instead of assuming every field is always present.

**Gemini's free-tier quota was much lower than expected.**
`gemini-2.5-flash` on the free tier turned out to be capped at 5 requests/minute and 20/day — got hit mid-eval-run with `RESOURCE_EXHAUSTED` errors on 9 out of 15 questions. Switched to `gemini-2.5-flash-lite`, which then turned out to be deprecated for new projects entirely (404, "no longer available to new users"). Ended up on `gemini-flash-lite-latest`, an alias that always points to the current lite-tier model instead of a hardcoded version that can go stale.

**Case-sensitive string matching silently breaking citation parsing.**
The generation step asks the model to report which chunks it used in a specific format (`USED_CHUNK_IDS: ...`). One response came back as `USED_CHUNK_IDs` (lowercase `s`) — an exact string match missed it, silently fell back to citing everything retrieved, and leaked raw formatting tokens into the answer text. Fixed by matching case-insensitively with a regex instead of an exact string check.

**Eval script timeouts on a fresh restart.**
After restarting both terminals from scratch, the first eval run kept timing out even with a 30s window. Bumped the timeout to 60s and added one automatic retry per question — cold starts and free-tier queuing can genuinely take that long sometimes, and it's a batch test script, not the live API, so the extra time doesn't matter here.

## What was hard

Debugging the rate-limit failures was the most confusing part, mainly because the API layer swallowed the real error and returned a generic 500 — had to fix the eval script to surface the actual response body before the real cause (a daily quota, not just a per-minute one) was even visible.

Keeping git history clean while iterating fast was harder than expected — a couple of fixes ended up bundled into earlier commits because testing happened before committing. Not a real problem in the end (nothing was lost), but a good reminder to commit right after each verified change, not after several changes at once.

## What was interesting

Watching the adversarial test cases work correctly was the most satisfying part — questions deliberately written to sound answerable (like asking about a security deposit in the employment agreement, when deposits only appear in the lease document) correctly got refused instead of the model blending unrelated documents together. Same with the two harder cases from Legixo's own gold test set — asking about a penalty that's never stated, and asking who won a case that's still pending. Both refused cleanly instead of inventing an answer, which is really the entire point of building this as a graph with an explicit "check before answering" step instead of one direct call to a model.
