#!/usr/bin/env python3
"""
Task 2: Text chunking, embedding, and vector-store indexing for complaint narratives.

Steps:
1) Load cleaned complaints from data/processed/filtered_complaints.csv.
2) Create a stratified sample (~12k by default) across product categories.
3) Chunk narratives with RecursiveCharacterTextSplitter (chunk_size=500, overlap=50).
4) Embed chunks with sentence-transformers/all-MiniLM-L6-v2.
5) Persist a ChromaDB collection under vector_store/chroma_task2.

Metadata stored with each chunk:
- complaint_id (if available)
- product_category (derived field)
- product (raw product string)
- issue (if present)
- chunk_index, total_chunks
"""
from pathlib import Path
from typing import List, Dict, Optional
import json
import math
import uuid

import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "filtered_complaints.csv"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "chroma_task2"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SAMPLE_SIZE = 12000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RANDOM_STATE = 42


def ensure_product_category(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a product_category column exists; derive from product column if missing."""
    df = df.copy()
    if "product_category" in df.columns:
        return df

    product_col = next((c for c in df.columns if "product" in c.lower()), None)
    if product_col is None:
        raise ValueError("No product column found to derive product_category")

    def categorize(product: str) -> str:
        p = str(product).lower()
        if "credit" in p and "card" in p:
            return "Credit Card"
        if "personal" in p and "loan" in p:
            return "Personal Loan"
        if "savings" in p and "account" in p:
            return "Savings Account"
        if "money" in p and "transfer" in p:
            return "Money Transfer"
        if "wire" in p and "transfer" in p:
            return "Money Transfer"
        return "Other"

    df["product_category"] = df[product_col].apply(categorize)
    return df


def stratified_sample(df: pd.DataFrame, sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Stratified sample across product_category, proportional with min 1 per stratum."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    counts = df["product_category"].value_counts()
    total = len(df)
    # Compute per-class counts proportional to distribution
    target_per_class = (counts / total * sample_size).round().astype(int)
    # Ensure at least one from each class present in data
    target_per_class = target_per_class.clip(lower=1)

    samples = []
    for category, n in target_per_class.items():
        subset = df[df["product_category"] == category]
        take = min(len(subset), n)
        samples.append(subset.sample(n=take, random_state=RANDOM_STATE))

    sampled_df = pd.concat(samples, axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled_df


def chunk_narrative(text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """Chunk a narrative string into overlapping segments."""
    if not isinstance(text, str) or not text.strip():
        return []
    return splitter.split_text(text)


def build_chunks(df: pd.DataFrame, text_col: str = "Consumer complaint narrative") -> List[Dict]:
    """Chunk narratives and attach metadata for each chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: List[Dict] = []

    if text_col not in df.columns:
        # fall back to any narrative-like column
        candidates = [c for c in df.columns if "narrative" in c.lower() or "complaint" in c.lower()]
        if not candidates:
            raise ValueError("No narrative column found")
        text_col = candidates[0]

    for _, row in df.iterrows():
        text = row[text_col]
        parts = chunk_narrative(text, splitter)
        total = len(parts)
        for idx, part in enumerate(parts):
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "id": chunk_id,
                "text": part,
                "metadata": {
                    "complaint_id": row.get("Complaint ID") or row.get("complaint_id"),
                    "product_category": row.get("product_category"),
                    "product": row.get("Product") or row.get("product"),
                    "issue": row.get("Issue") or row.get("issue"),
                    "chunk_index": idx,
                    "total_chunks": total,
                }
            })
    return chunks


def embed_chunks(chunks: List[Dict], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Embed chunk texts in batches and return numpy array."""
    texts = [c["text"] for c in chunks]
    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings)


def persist_chroma(chunks: List[Dict], embeddings: np.ndarray, collection_name: str = "task2_chunks") -> None:
    """Persist chunks + embeddings into ChromaDB at VECTOR_STORE_DIR."""
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    # recreate collection to avoid duplicates on rerun
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(name=collection_name, metadata={"source": "task2"})

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())

    # Save a small manifest for traceability
    manifest = {
        "collection": collection_name,
        "num_chunks": len(chunks),
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(VECTOR_STORE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main(sample_size: int = DEFAULT_SAMPLE_SIZE) -> None:
    print("=== Task 2: Chunk, Embed, and Index ===")
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed file not found at {PROCESSED_PATH}")

    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded processed data: {df.shape}")

    df = ensure_product_category(df)
    sample_df = stratified_sample(df, sample_size=sample_size)
    print(f"Sampled {len(sample_df):,} complaints across products: \n{sample_df['product_category'].value_counts()}")

    chunks = build_chunks(sample_df)
    print(f"Built {len(chunks):,} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_chunks(chunks, model)
    print(f"Computed embeddings shape: {embeddings.shape}")

    persist_chroma(chunks, embeddings)
    print(f"Persisted vector store to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
