def build_rag_prompt(question: str, context_chunks) -> str:
    """Build a standard RAG prompt without adaptive routing logic."""
    if isinstance(context_chunks, dict):
        memory_chunks = context_chunks.get("memory_context", [])
        evidence_chunks = context_chunks.get("external_evidence", [])
        sections = []
        if memory_chunks:
            sections.append(
                "Temporary conversational memory:\n"
                + "\n\n".join(_context_text(chunk) for chunk in memory_chunks)
            )
        if evidence_chunks:
            sections.append(
                "External evidence:\n"
                + "\n\n".join(_context_text(chunk) for chunk in evidence_chunks)
            )
        context_text = "\n\n".join(sections)
    elif isinstance(context_chunks, str):
        context_text = context_chunks
    else:
        context_text = "\n\n".join(
            f"[Document {idx + 1}] {chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)}"
            for idx, chunk in enumerate(context_chunks)
        )

    return (
        "Answer the question using the provided context.\n"
        "If the context does not contain enough information, say that the information is insufficient.\n"
        "Do not invent facts.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}"
    )


def _context_text(chunk) -> str:
    if not isinstance(chunk, dict):
        return str(chunk)
    if "original_query" in chunk:
        return (
            f"Previous user query: {chunk.get('original_query', '')}\n"
            f"Previous assistant response: {chunk.get('stored_response', '')}\n"
            f"Previous context: {chunk.get('stored_context', '')}"
        )
    text = chunk.get("text") or chunk.get("stored_context") or chunk.get("stored_response") or chunk.get("original_query")
    return str(text or chunk)
