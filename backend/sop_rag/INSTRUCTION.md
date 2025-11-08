# SOP RAG System

## Overview

The SOP RAG system provides an end-to-end pipeline for:

1. **PDF Content Extraction**: Intelligently parses PDF documents to extract text, tables, and images
2. **Content Summarization**: Uses Large Language Models (LLMs) to create concise summaries of extracted content
3. **Vector Storage**: Stores content summaries in a vector database for semantic search
4. **Question Answering**: Provides a RAG-based interface for querying the processed documents

## Architecture

```
PDF Document → Content Extraction → Content Separation → Summarization → Vector Storage → RAG Query Interface
```

## Usage

### Indexer Phase
- Step 1: Put necessary documents in data folder (see config.py)
- Step 2: Run python sop_indexer.py

### Retriever Phase
- Step 1: Run python sop_retriever.py.
