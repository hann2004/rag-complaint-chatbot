#!/usr/bin/env python3
"""
Task 4: Interactive RAG Chat Interface (Streamlit)

Features:
- User question input and Ask button.
- Displays generated answer and retrieved source chunks for transparency.
- Clear button to reset the conversation.
- Optional pseudo-streaming of the final answer for better UX.

Environment:
- Uses the persisted Chroma store at vector_store/chroma_task2 (built in Task 2).
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (same as Task 2).
- Generator model selectable via UI; defaults to TinyLlama chat if available.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# Paths and defaults
PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "chroma_task2"
COLLECTION_NAME = "task2_chunks"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ALT_LLM = "google/flan-t5-base"


@st.cache_resource(show_spinner=False)
def load_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        return client.get_collection(COLLECTION_NAME)
    return client.create_collection(name=COLLECTION_NAME, metadata={"source": "task4"})


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    task = "text2text-generation" if "flan" in model_name.lower() or "t5" in model_name.lower() else "text-generation"
    return pipeline(task, model=model_name)


def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    embedder = load_embedder()
    collection = load_collection()
    q_emb = embedder.encode([question], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]
    res = collection.query(query_embeddings=[q_emb.tolist()], n_results=int(top_k), include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    results = []
    for i, doc in enumerate(docs):
        results.append({
            "text": doc,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return results


def build_prompt(chunks: List[str], question: str) -> str:
    truncated = [c[:300] for c in chunks[:2]]
    numbered = [f"[{i+1}] {c}" for i, c in enumerate(truncated)]
    context = "\n\n".join(numbered)
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Answer using ONLY the provided excerpts. "
        "Rephrase in your own words in 1-2 sentences (<=60 words). "
        "Avoid copying long phrases, avoid repetition and lists. "
        "If the context lacks the answer, say you don't have enough information. "
        "Cite sources with bracket numbers and include complaint_id.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (with source numbers):"
    )
    return prompt


def generate(prompt: str, model_name: str) -> str:
    gen = load_generator(model_name)
    task = gen.task
    if task == "text2text-generation":
        out = gen(prompt, max_new_tokens=64)
        return out[0]["generated_text"].strip()
    out = gen(
        prompt,
        max_new_tokens=64,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
    )
    text = out[0]["generated_text"]
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def stream_text(text: str):
    for token in text.split():
        yield token + " "
        time.sleep(0.01)


def render_sources(retrieved: List[Dict[str, Any]]):
    if not retrieved:
        st.info("No sources retrieved.")
        return
    for idx, item in enumerate(retrieved[:3], start=1):
        meta = item.get("metadata", {}) or {}
        cid = meta.get("complaint_id", "-")
        cat = meta.get("product_category", "-")
        st.markdown(f"**Source {idx}** — {cat}, complaint_id={cid}")
        st.write(item.get("text", ""))
        st.caption(f"Distance: {item.get('distance')}")
        st.divider()


def run_query(question: str, top_k: int, model_name: str) -> Dict[str, Any]:
    retrieved = retrieve(question, top_k=top_k)
    chunks = [r["text"] for r in retrieved]
    prompt = build_prompt(chunks, question)
    answer = generate(prompt, model_name=model_name)

    citations = []
    for i, r in enumerate(retrieved[:2], start=1):
        meta = r.get("metadata", {}) or {}
        citations.append(f"[{i}] {meta.get('product_category', '-')}, complaint_id={meta.get('complaint_id', '-')}")
    if citations:
        answer = answer + "\n\nCitations: " + "; ".join(citations)

    return {"answer": answer, "retrieved": retrieved}


def main():
    st.set_page_config(page_title="CrediTrust RAG Chat", layout="wide")
    # Subtle theming tweak: use a calmer accent color instead of red
    st.markdown(
        """
        <style>
        :root {
            --accent-color: #2f80ed;
            --accent-hover: #1c64c7;
            --border-color: #e6eef8;
        }
        /* Primary buttons (Ask) */
        .stButton button[kind="primary"] {
            background: var(--accent-color);
            color: white;
            border: 1px solid var(--accent-color);
        }
        .stButton button[kind="primary"]:hover {
            background: var(--accent-hover);
            border-color: var(--accent-hover);
        }
        /* Secondary buttons (Clear) */
        .stButton button:not([kind="primary"]) {
            border: 1px solid var(--border-color);
            color: #1a1a1a;
            background: white;
        }
        .stButton button:not([kind="primary"]):hover {
            border-color: var(--accent-color);
            color: var(--accent-color);
        }
        /* Text area focus ring */
        textarea:focus, textarea:focus-visible {
            outline: 2px solid var(--accent-color) !important;
            box-shadow: 0 0 0 1px var(--accent-color) !important;
            border-color: var(--accent-color) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("CrediTrust RAG Chat")
    st.write("Ask a question about customer complaints. The app retrieves relevant chunks and generates an answer with citations.")

    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("Generator model", [DEFAULT_LLM, ALT_LLM], index=0)
        top_k = st.slider("Top-k chunks", min_value=1, max_value=5, value=1, step=1)
        streaming = st.checkbox("Stream answer", value=True)
        st.caption("Models load on first use and may take time.")

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "clear_input_next" not in st.session_state:
        st.session_state.clear_input_next = False

    # Clear the input before rendering the widget if flagged
    if st.session_state.clear_input_next:
        st.session_state.question_input = ""
        st.session_state.clear_input_next = False

    question = st.text_area(
        "Your question",
        placeholder="e.g., What issues are common in credit card disputes?",
        height=120,
        key="question_input",
    )

    col1, col2 = st.columns([1, 1])
    ask = col1.button("Ask", type="primary")
    clear = col2.button("Clear")

    # Conversation state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of dicts: {question, answer, retrieved}

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Retrieving and generating..."):
            try:
                result = run_query(question.strip(), top_k=top_k, model_name=model_choice)
            except Exception as exc:  # pragma: no cover
                st.error(f"Error: {exc}")
                return

        # Append to history
        st.session_state.chat_history.append({"question": question.strip(), "answer": result["answer"], "retrieved": result["retrieved"]})
        # Flag clearing the input for the next render and rerun
        st.session_state.clear_input_next = True
        st.rerun()

    # Render chat history (most recent last)
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for turn in st.session_state.chat_history:
            st.markdown(f"**You:** {turn['question']}")
            if streaming:
                st.write_stream(stream_text(turn["answer"]))
            else:
                st.write(turn["answer"])
            st.markdown("**Sources:**")
            render_sources(turn["retrieved"])
            st.divider()


if __name__ == "__main__":
    main()
