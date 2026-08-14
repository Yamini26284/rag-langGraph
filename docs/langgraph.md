# How the question-answering flow works

When someone asks a question, it doesn't go straight to an AI model and back. It moves through a small series of steps, and at one point the flow actually splits into two different paths depending on what was found. Here's each step, in plain terms.

## The steps

**1. Search**
The question is turned into a search query and matched against the stored document pieces. The 4 closest matches come back, each with a score showing how close a match it is.

**2. Check**
This is the important step. Before trusting what was found, it asks: *is this actually good enough to answer the question?*

- If the match is clearly strong, it says yes right away.
- If the match is clearly weak, it says no right away.
- If it's somewhere in between, it does one more check, it actually reads the passages and asks a model "does this genuinely answer the question, or did it just look close?" This extra check only happens for the unclear cases, so it's not wasted effort on the obvious ones.

Either way, it always writes down *why* it made that decision, not just yes or no.

**3a. Write the answer** (if the check said yes)
An answer is written using only the passages that were found and nothing outside them. Every source used gets listed alongside the answer, so it's clear exactly where each part of the answer came from.

**3b. Say "can't find it"** (if the check said no)
If the passages weren't good enough, it doesn't guess. It tries searching again, up to 2 times, in case the first attempt just didn't hit the right spot. If it still can't find anything solid after that, it plainly says the documents don't have the answer.

## The branch and the safety limit

This is the part that makes it a real flow instead of one big block of logic:

- After the "Check" step, there's a genuine fork so good matches go one way, weak matches go another.
- If it takes the weak path, it's allowed to retry, but only a limited number of times (2). Once that limit is hit, it's forced to stop and say it can't find the answer, instead of ever searching forever.

## Diagram

```
question
   │
   ▼
 search
   │
   ▼
 check ──────────────┐
   │                 │
  good               weak
   │                 │
   ▼                 ▼
write answer     tried twice already?
                       │
              ┌────────┴────────┐
             no                yes
              │                 │
              ▼                 ▼
        search again       say "can't find it"
        (back to check)
```

## Why it's built this way

Splitting search, checking, and answering into separate steps (instead of one big request to an AI model) means each part can be looked at on its own — you can see exactly what was found, why it was trusted or not, and what it was based on. That's also what makes the honest "I can't find this" responses possible instead of the system just making something up when it's unsure.

## Where each step lives in the code

| Step above | Function in `app/graph.py` |
|---|---|
| Search | `retrieve_node` |
| Check | `grade_node` |
| The fork (good vs. weak) | `retry_router` |
| Write the answer | `generate_node` |
| Search again | `bump_retry_node` (raises the retry count, then loops back to `retrieve_node`) |
| Say "can't find it" | `refuse_node` |
| Retry limit | `MAX_RETRIES` constant, checked inside `retry_router` |
