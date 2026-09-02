# Local App Guidelines

The full script for local version is available on [github.com/pharmbio/repuragent](https://github.com/pharmbio/repuragent). Clone it and run it with Docker. With the Local app, everything will stay on your device, including running processes, input/output files, and conversation history.

## 2.1 Prerequisites

| Requirement | Details |
| --- | --- |
| `OPENAI_API_KEY` | Mandatory. To audit OpenAI models |
| Docker Desktop | Mandatory. To run Docker Compose. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/). |
| LangSmith | Optional. To trace the run in detail. More details on [smith.langchain.com](https://smith.langchain.com)


## 2.2 Quick Start

1. **Clone or update the repo**
   ```bash
   git clone https://github.com/pharmbio/repuragent.git
   cd repuragent
   ```
2. **Create `.env`**
   ```bash
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
   ```
3. **Build and run**
   ```bash
   docker-compose up --build
   ```

4. **Access the UI**  
   Visit [http://localhost:7860](http://localhost:7860) and follow the
   [Usage Guidelines](shared_usage.md) to start working with the agent.


5. **Stop/restart**  
   Use `docker-compose down` to stop. And `docker-compose up -d` to resume the agent whenever you need.

## 2.3 Optional settings

### LangSmith Tracing
1. Create a project at [smith.langchain.com](https://smith.langchain.com) and generate an API key.
2. Append to `.env`:
   ```bash
   echo "LANGCHAIN_TRACING_V2=true"  > .env
   echo "LANGCHAIN_ENDPOINT=https://api.smith.langchain.com" > .env
   echo "LANGCHAIN_API_KEY=lsm-your-langsmith-key" > .env
   echo "LANGCHAIN_PROJECT=your-project-name" > .env
   ```
3. Restart Docker. Every run now emits detailed traces (graph nodes, tool calls, token usage) to the LangSmith dashboard.

## 2.4 Injecting New SOPs (Local App Only)

The local version lets you inject your own SOP documents into the system.

1. **Drop source files** into `persistence/data/SOP/`. Keep filenames descriptive: the
   filename is written into the indexed text, so a document is findable by the number a
   person would cite even when that number appears nowhere in its body.
2. **Index or re-index** with the provided script:
   ```bash
   cd persistence/data/SOP
   python reindex.py               # index whatever is new, edited or deleted
   python reindex.py --dry-run     # say what would change, write nothing
   python reindex.py --rebuild     # discard the index and start again
   ```
   - The indexer parses each PDF into sections, embeds ~200-character children of them into ChromaDB (`persistence/memory/sop_documents/ensemble/chroma_db`), keeps the whole sections in a docstore (`.../ensemble/docstore`), and writes the keyword arm's corpus to `.../ensemble/bm25_corpus.json`. It is incremental: a SHA-256 per file in `manifest.json` means only what is new or edited is re-parsed.

3. **How it becomes available**  
   The retriever is built on first use and cached, loading the persisted vector store, the
   parent docstore and the keyword corpus. An agent calling `protocol_search_sop` then
   searches your newest documents alongside the shipped ones. Re-index while the app is
   running and restart it, or the cached retriever will still hold the previous corpus.

## 2.5 Local Data Handling

- **Everything is local**  
  Uploads, results, embeddings, checkpoints, and logs stay under the repo folder you
  cloned. No data is sent out.
- **No hidden uploads**  
  The app never transmits your data except to the model endpoints you configure (OpenAI,
  LangSmith). Review their own privacy terms—the local app relays your prompts/files
  over the API calls your key authorises.

With the prerequisites in place, you can rely on this guide plus the [Usage Guidelines](shared_usage.md) to operate Repuragent locally.