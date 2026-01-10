# rag-complaint-chatbot

[![CI](https://github.com/hann2004/rag-complaint-chatbot/actions/workflows/unittests.yml/badge.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue?logo=postgresql)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Internal RAG assistant for CrediTrust complaint analysis. Task 1 covers EDA/cleaning; Task 2 chunks, embeds, and indexes narratives for semantic search; Task 3 builds the retrieval + generation pipeline and evaluation; Task 4 provides an interactive chat UI.

## Project structure
- `src/task1_processing.py`: EDA + preprocessing (filter target products, clean narratives, save filtered CSV, EDA report, notebook generator).
- `src/task2_chunk_embed.py`: Stratified sample, chunking (500/50), MiniLM embeddings, Chroma vector store persisted to `vector_store/chroma_task2`.
- `src/task3_rag.py`: RAG pipeline (retriever + prompt + generator) and qualitative evaluation report writer.
- `app.py`: Streamlit chat interface for non-technical users.
- `data/processed/`: Filtered complaints, EDA sample, EDA report (ignored by git).
- `vector_store/`: Persisted vector DB artifacts (ignored by git).
- `docs/task2_report.md`: Sampling, chunking, embedding choices and how to run Task 2.
- `docs/task3_report.md`: RAG evaluation table (questions, answers, retrieved sources, scores/comments). Add your screenshots/GIF under `docs/images/` and reference here.

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

## Run Task 3 (RAG pipeline + evaluation)
The evaluation runs a set of questions, retrieves top-k chunks from the persisted Chroma store, prompts an instruction model, and writes a Markdown table to `docs/task3_report.md`.

```bash
# Choose an instruct model (env var optional; defaults to TinyLlama chat)
RAG_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 python -m src.task3_rag
# Alternatives: google/flan-t5-base or flan-t5-small (smaller, CPU-friendly)
```

Outputs:
- `docs/task3_report.md`: Questions, answers, retrieved sources (1–2), score placeholder, and comments.

Notes:
- The retriever uses the same embedding model as Task 2 (`all-MiniLM-L6-v2`) and loads `vector_store/chroma_task2`.
- The prompt asks for concise, grounded answers and includes source citations.

## Run Task 4 (Interactive Streamlit app)
Launch the chat UI that wraps the same retriever + generator and reveals source chunks for trust.

```bash
streamlit run app.py
```

Features:
- Text input and Ask button, Clear to reset conversation.
- Displays answer and top sources with `complaint_id` and `product_category`.
- Optional token-like streaming for answers.
- Sidebar settings: model choice (TinyLlama or FLAN-T5) and top-k (defaults to 1 for focused answers).

Tip: First invocation will download the selected model; allow time or pre-warm by setting `RAG_LLM_MODEL`.

## Capture a demo GIF (optional)
Record a short demo of the app and embed it into the report.

```bash
# Ubuntu/Debian (Peek GUI)
sudo apt update && sudo apt install peek

# Record and save to docs/images/creditrust_demo.gif, then embed:
```

Add to `docs/task3_report.md`:

```markdown
![CrediTrust RAG Chat Demo](images/creditrust_demo.gif)
```
## Testing
```bash
pytest
```

## Notes
- Large data and vector artifacts are git-ignored (`data/raw`, `data/processed`, `vector_store`).
- Badge links assume GitHub Actions workflow at `.github/workflows/unittests.yml`.