# Module 3: Adaptive Query Router (AQR)

This module implements the first CARO orchestration step: routing an incoming query to a high-level decision before any retrieval or generation occurs.

## Purpose

The Adaptive Query Router does not perform retrieval or invoke an LLM. Instead, it analyzes the query and emits a deterministic, explainable routing decision:

- DIRECT
- RETRIEVE
- MEMORY
- RETRIEVE_AND_MEMORY

This keeps the routing layer independent from the baseline RAG system in Module 2 and makes it easy to extend in later CARO modules.

## Why routing happens before retrieval

The router provides a structured decision before the downstream system decides whether it needs external knowledge, memory, or both. This keeps the pipeline explainable and allows future modules to plug in different retrieval strategies without rewriting the routing logic.

## Heuristic design

The router uses lightweight deterministic heuristics such as:

- query length
- presence of follow-up references like "this", "that", "earlier", "previously"
- procedural cues like "how do I" or "how to"
- comparison cues like "difference between"
- multi-hop phrasing such as "person who founded"

The confidence score is intentionally heuristic rather than statistical. It is
a rule-strength score, not a calibrated probability and not an ML classifier
confidence.

## Decision process

Module 3 asks: **What does the query need?** It independently determines the
question type, whether external retrieval is required, and whether
conversational memory is required.

The route follows this deterministic mapping:

| Retrieval | Memory | Route |
| --- | --- | --- |
| false | false | `DIRECT` |
| true | false | `RETRIEVE` |
| false | true | `MEMORY` |
| true | true | `RETRIEVE_AND_MEMORY` |

The result includes signals derived from the analyzer, including procedural,
comparison, multi-hop, contextual-reference, retrieval-required, and
memory-required signals. These signals explain the decision; they do not imply
that a named entity or a complex question automatically requires retrieval or
memory. Complexity/question type does not automatically imply memory.

Examples:

- `What is the difference between REST and SOAP?` -> comparison, retrieval required, memory not required, `RETRIEVE`.
- `Which country was the birthplace of the person who founded the company?` -> multi-hop, retrieval required, memory not required, `RETRIEVE`.
- `What about its advantages?` -> conversational, memory required, retrieval not currently required, `MEMORY`.

Reasons describe the detected characteristics and rule path. They use careful
language such as "under the current routing rules" and "no explicit retrieval
requirement was detected" because these are deterministic heuristics, not a
mathematical proof that retrieval is unnecessary.

## Why MEMORY is present even without a real memory system

The routing layer identifies when a query likely depends on prior context or conversational state, even though the actual memory implementation is intentionally not built in this module. This keeps the contract stable for future Modules 4-8.

## Future integration

Later CARO modules can consume the structured output of the router and decide how to trigger retrieval, memory lookup, or both. This module does not implement those behaviors yet.
