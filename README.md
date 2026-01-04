# rag-complaint-chatbot

[![CI](https://github.com/hann2004/rag-complaint-chatbot/actions/workflows/unittests.yml/badge.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue?logo=postgresql)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Internal RAG assistant for CrediTrust complaint analysis. Task 1 covers EDA/cleaning; Task 2 chunks, embeds, and indexes narratives for semantic search.

## Project structure
- `src/task1_processing.py`: EDA + preprocessing (filter target products, clean narratives, save filtered CSV, EDA report, notebook generator).
- `src/task2_chunk_embed.py`: Stratified sample, chunking (500/50), MiniLM embeddings, Chroma vector store persisted to `vector_store/chroma_task2`.
- `data/processed/`: Filtered complaints, EDA sample, EDA report (ignored by git).
- `vector_store/`: Persisted vector DB artifacts (ignored by git).
- `docs/task2_report.md`: Sampling, chunking, embedding choices and how to run Task 2.

## Setup
```bash
pip install -r requirements.txt
```
Place the CFPB dataset at `data/raw/complaints.csv`.

## Run Task 1 (EDA & cleaning)
```bash
python run_task1.py
```
Outputs (in `data/processed/` and `notebooks/`):
- `filtered_complaints.csv`
- `complaints_sample_eda.csv`
- `eda_report.txt`
- `task1_eda.ipynb`

## Run Task 2 (chunk, embed, index)
```bash
python src/task2_chunk_embed.py
```
Defaults: stratified sample (~12k target; uses all if smaller), chunk_size=500, chunk_overlap=50, model=all-MiniLM-L6-v2. Persists Chroma DB to `vector_store/chroma_task2` with a manifest.

## Testing
```bash
pytest
```

## Notes
- Large data and vector artifacts are git-ignored (`data/raw`, `data/processed`, `vector_store`).
- Badge links assume GitHub Actions workflow at `.github/workflows/unittests.yml`.