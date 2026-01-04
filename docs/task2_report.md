# Task 2: Chunking, Embedding, and Indexing

## Sampling strategy (10–15k)
- Stratified sample target: 12,000 rows (`DEFAULT_SAMPLE_SIZE`), proportional to `product_category` distribution.
- Each category gets at least 1 sample; counts rounded from class proportions.
- Source: `data/processed/filtered_complaints.csv` (cleaned in Task 1).

## Chunking approach
- Splitter: `RecursiveCharacterTextSplitter` (langchain-text-splitters).
- Parameters: `chunk_size=500`, `chunk_overlap=50`.
- Rationale: 500 characters keeps narratives concise for MiniLM while overlap preserves context across boundaries; light enough for fast embedding.

## Embedding model choice
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, lightweight, strong general-purpose sentence embeddings, fast CPU-friendly).
- Normalized embeddings used for cosine similarity.

## Indexing
- Vector store: Chroma persistent collection at `vector_store/chroma_task2`.
- Stored fields per chunk: text, `complaint_id`, `product_category`, `product`, `issue`, `chunk_index`, `total_chunks`.
- Manifest saved to `vector_store/chroma_task2/manifest.json` for traceability.

## How to run
```bash
python src/task2_chunk_embed.py  # uses defaults: 12k sample, 500/50 chunking, MiniLM
```

## Outputs
- Chroma collection persisted under `vector_store/chroma_task2/` (ignored by git).
- Manifest summarizing model and chunking parameters.
