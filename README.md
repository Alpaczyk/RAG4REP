# RAG4REP — RAG for Repositories

A RAG pipeline that answers: **"which files in this GitHub repository are relevant to my question?"** — retrieval over a repository's files rather than plain text chunks.

## How it works

1. **Index** — given a repository URL, fetch its file tree from the GitHub API and summarize each file with an LLM
2. **Embed** — summarize and embed with `sentence-transformers` (all-MiniLM-L6-v2), indexed in **ChromaDB**
3. **Retrieve** — **multi-query expansion**: an LLM generates related questions for the user query; all are embedded and used to retrieve the most relevant file names
4. **Evaluate** — recall over an LLM-generated evaluation set of (question → relevant files) pairs

## Usage

```bash
pip install -r requirements.txt   # openai, sentence-transformers, chromadb, langchain-text-splitters, pandas
export OPENAI_API_KEY=...
python Run.py
```

Configure the target repository inside `Run.py`.

> Status: research prototype — the indexing and evaluation steps are still being iterated on (`Pipeline.py`, `Indexing.py`, `Evaluator.py`).

## Stack

Python · OpenAI API · sentence-transformers · ChromaDB · langchain · pandas
