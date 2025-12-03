# Local App Guidelines

The local version is the exact code that lives on [github.com/pharmbio/repuragent](https://github.com/pharmbio/repuragent). Clone it, run it with Docker, and everything—uploads, conversation history, model artifacts—stays on your machine.

## 2.1 Prerequisites

| Requirement | Details |
| --- | --- |
| `OPENAI_API_KEY` | Mandatory. The key never leaves your workstation; it is read from `.env` by `app/config.py`. |
| Docker Desktop | Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/). |
| Optional | LangSmith account and API key for tracing task details. More details on [smith.langchain.com](https://smith.langchain.com)


## 2.2 Quick Start

1. **Clone or update the repo.**
   ```bash
   git clone https://github.com/pharmbio/repuragent.git
   cd repuragent
   ```
2. **Create `.env`.**
   ```bash
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
   ```
3. **Build and run.**
   ```bash
   docker-compose up --build
   ```

4. **Access the UI**  
   Visit [http://localhost:7860](http://localhost:7860) and follow the
   [Shared Usage Guidelines](shared_usage.md) to start prompting.

5. **Stop/restart**  
   Use `docker-compose down` and `docker-compose up -d` whenever you need.

## 2.3 Optional Integrations

### LangSmith Tracing
1. Create a project at [smith.langchain.com](https://smith.langchain.com) and generate an API key.
2. Append to `.env`:
   ```bash
   echo "LANGCHAIN_TRACING_V2=true"  > .env
   echo "LANGCHAIN_ENDPOINT=https://api.smith.langchain.com" > .env
   echo "LANGCHAIN_API_KEY=lsm-your-langsmith-key" > .env
   echo "LANGCHAIN_PROJECT=repuragent-local" > .env
   ```
3. Restart Docker. Every run now emits detailed traces (graph nodes, tool calls, token usage) to the LangSmith dashboard.

### Debugging & Logs
- **Real-time logs**  
  `docker-compose logs -f repuragent`.
- **Hot-reload code**  
  Edit files on the host, then re-run `docker-compose up --build` to rebuild the image with your changes.

## 2.4 Injecting New SOPs (Local App Only)

The local version lets you inject your own SOP documents into the system

1. **Drop source files** into `data/SOP/`. The SOP indexer accepts PDF and DOCX; keep filenames descriptive (they appear in citations).
2. **Index or re-index** with the provided script:
   ```bash
   cd backend/sop_rag
   python sop_indexer.py
   ```
   - The script chunks documents, stores summaries in ChromaDB (`backend/memory/sop_documents/chromadb/sop_rag`), and saves original pages in a docstore folder (`backend/memory/sop_documents/docstore`).

3. **How it becomes available**  
   On app startup, `SOPRetriever` loads the persisted vector store. When an agent calls
   `protocol_search_sop`, it retrieves your newest snippets automatically.

4. **Refreshing content**  
   Repeat the indexer any time you add/edit SOP files. Delete
   `backend/memory/episodic_memory/chroma_db` if you need a clean rebuild SOP library.

## 2.5 Local Data Handling

- **Everything is local**  
  Uploads, results, embeddings, checkpoints, and logs stay under the repo folder you
  cloned. No telemetry is sent unless you explicitly enable LangSmith.
- **No hidden uploads**  
  The app never transmits your data except to the model endpoints you configure (OpenAI,
  LangSmith). Review their own privacy terms—the local app simply relays your prompts/files
  over the API calls your key authorizes.

With the prerequisites in place, you can rely on this guide plus the [Shared Usage Guidelines](shared_usage.md) to operate Repuragent completely locally.
