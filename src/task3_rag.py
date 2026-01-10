#!/usr/bin/env python3
"""
Task 3: Retrieval-Augmented Generation (RAG) core logic and evaluation.

Functions:
- load_vector_store(): Load Chroma persistent collection built in Task 2.
- embed_query(text): Embed a query using all-MiniLM-L6-v2 (same as Task 2).
- retrieve_top_k(query, k): Similarity search returning top-k chunks.
- build_prompt(context, question): Prompt template guiding the LLM.
- generate_answer(prompt, model_name): Use Hugging Face text-generation pipeline.
- answer_question(question, k, model_name): Full RAG pipeline for a single question.
- run_evaluation(questions, k, model_name, output_path): Run qualitative eval and save Markdown report.

CLI:
Running as a module will execute a default evaluation and write docs/task3_report.md.
Optionally set env var RAG_LLM_MODEL to choose the HF model (default: sshleifer/tiny-gpt2).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "chroma_task2"
COLLECTION_NAME = "task2_chunks"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_vector_store() -> "chromadb.api.models.Collection.Collection":
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    # Use existing collection; create if missing (empty)
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        return client.get_collection(COLLECTION_NAME)
    # Fallback: create empty collection (to avoid hard crash)
    return client.create_collection(name=COLLECTION_NAME, metadata={"source": "task3"})


def embed_query(text: str, model: SentenceTransformer) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Question must be a non-empty string")
    emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return emb[0]


def retrieve_top_k(question: str, k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
    collection = load_vector_store()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = embed_query(question, model)

    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=int(k),
        include=["documents", "metadatas", "distances"],
    )
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    return documents, metadatas, distances


def build_prompt(context_chunks: List[str], question: str) -> str:
    # Truncate each chunk to keep within model context limits
    truncated = [chunk[:500] for chunk in context_chunks]
    numbered = [f"[{i+1}] {chunk}" for i, chunk in enumerate(truncated)]
    context = "\n\n".join(numbered)
    template = (
        "You are a financial analyst assistant for CrediTrust. "
        "Use ONLY the provided complaint excerpts to answer. "
        "Respond concisely in 2-4 sentences. "
        "If the context lacks the answer, say you don't have enough information. "
        "Cite sources using their bracket numbers and include complaint_id when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (with source numbers):"
    )
    return template


def generate_answer(prompt: str, model_name: str | None = None) -> str:
    # Prefer a small instruction-tuned chat model when not specified
    default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = model_name or os.environ.get("RAG_LLM_MODEL", default_model)

    # Choose pipeline type based on model; FLAN-T5 uses text2text-generation
    task = "text2text-generation" if "flan" in model_name.lower() or "t5" in model_name.lower() else "text-generation"
    gen = pipeline(task, model=model_name)

    if task == "text2text-generation":
        out = gen(prompt, max_new_tokens=128)
        text = out[0]["generated_text"].strip()
        return text
    else:
        out = gen(prompt, max_new_tokens=128, temperature=0.3, do_sample=True)
        text = out[0]["generated_text"]
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()


def answer_question(question: str, k: int = 5, model_name: str | None = None) -> Dict:
    docs, metas, dists = retrieve_top_k(question, k=k)
    if not docs:
        answer = "I don't have enough information to answer from the context."
    else:
        prompt = build_prompt(docs, question)
        answer = generate_answer(prompt, model_name=model_name)

    retrieved = [
        {"text": docs[i], "metadata": metas[i], "distance": dists[i]}
        for i in range(len(docs))
    ]

    # Append brief citations (top 2) to the answer
    citations = []
    for idx, r in enumerate(retrieved[:2], start=1):
        m = r.get("metadata", {}) or {}
        cid = m.get("complaint_id", "-")
        cat = m.get("product_category", "-")
        citations.append(f"[{idx}] {cat}, complaint_id={cid}")
    if citations:
        answer = answer + "\n\nCitations: " + "; ".join(citations)

    return {"question": question, "answer": answer, "retrieved": retrieved}


def _default_questions() -> List[str]:
    return [
        "What common issues are reported with credit card disputes?",
        "Are there frequent complaints about wire transfer delays?",
        "What problems do customers face with savings accounts?",
        "Do personal loan customers mention billing errors or fees?",
        "How often do money transfer complaints mention failed transactions?",
        "Is there evidence of unauthorized charges in complaints?",
        "What resolution times are described for disputes?",
    ]


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def run_evaluation(
    questions: List[str] | None = None,
    k: int = 5,
    model_name: str | None = None,
    output_path: Path | None = None,
) -> Path:
    questions = questions or _default_questions()
    output_path = output_path or (PROJECT_ROOT / "docs" / "task3_report.md")

    rows = []
    for q in questions:
        result = answer_question(q, k=k, model_name=model_name)
        # Show 1-2 sources
        sources = result["retrieved"][:2]
        src_strs = []
        for s in sources:
            meta = s.get("metadata", {}) or {}
            prefix = f"[{meta.get('product_category', 'Unknown')}] "
            snippet = s.get("text", "")[:200]
            src_strs.append(prefix + snippet)
        rows.append({
            "question": q,
            "answer": result["answer"],
            "sources": src_strs,
            # Placeholder score and comments; manual review recommended
            "score": "-",
            "comments": "Review needed",
        })

    # Build Markdown table
    header = (
        "# Task 3: RAG Evaluation\n\n"
        "This table summarizes qualitative evaluation of the RAG pipeline. "
        "Scores and comments should be refined after manual review.\n\n"
        "| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |\n"
        "|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        sources_joined = "<br/>".join(_md_escape(s) for s in r["sources"]) if r["sources"] else ""
        line = (
            f"| {_md_escape(r['question'])} | {_md_escape(r['answer'])} | {sources_joined} | {r['score']} | {_md_escape(r['comments'])} |\n"
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_path


def main():
    model_name = os.environ.get("RAG_LLM_MODEL", "sshleifer/tiny-gpt2")
    out = run_evaluation(model_name=model_name)
    print(f"Wrote evaluation report to {out}")


if __name__ == "__main__":
    main()
