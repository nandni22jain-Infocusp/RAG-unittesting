# RAG-unittesting

This repository demonstrates a **basic Retrieval-Augmented Generation (RAG) pipeline** using Hugging Face models and the [LlamaIndex](https://github.com/run-llama/llama_index) library, along with comprehensive unit tests for all major components.

## Features

- Loads documents from a Hugging Face dataset (e.g., SQuAD)
- Embeds documents and builds a vector index
- Retrieves relevant documents for a query
- Generates answers using a language model
- Includes robust unit and integration tests for the RAG workflow

## Structure

- `main.py` — Example script to run the RAG pipeline
- `rag_system.py` — Core RAG implementation
- `test_ragsystem.py` — Unit and integration tests (using `unittest` and `unittest.mock`)
- `requirements.txt` — Python dependencies

## Quick Start

1. **Install dependencies:**
```python
pip install -r requirements.txt
```


2. **Set your OpenAI API key** in a `.env` file:
```python
OPENAI_API_KEY=your_openai_api_key
```


3. **Run the example:**
```python
python main.py
```


4. **Run all tests:**
```python
python test_ragsystem.py
```


## Notes

- The RAG system uses Hugging Face models for embeddings and LLMs, and OpenAI for LLM inference.
- All major RAG steps are covered by unit tests, including error handling.

---

**For learning and experimentation.**