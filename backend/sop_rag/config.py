import os
from pathlib import Path

# Base paths - now works from project root where experiments.ipynb is located
BASE_DIR = Path(__file__).parent.parent.parent  # Project root (where experiments.ipynb is)
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = BASE_DIR / "backend" / "memory"

# SOP specific paths
SOP_DATA_DIR = DATA_DIR / "SOP"
SOP_MEMORY_DIR = MEMORY_DIR / "sop_documents"
CHROMA_PERSIST_PATH = SOP_MEMORY_DIR / "chroma_db" / "sop_rag"
DOCSTORE_PATH = SOP_MEMORY_DIR / "docstore.pkl"

# ChromaDB configuration
COLLECTION_NAME = 'sop_rag'
ID_KEY = 'doc_id'

# PDF processing configuration
PDF_PROCESSING_CONFIG = {
    'strategy': 'hi_res',
    'infer_table_structure': True,
    'extract_image_block_types': ['Image'],
    'extract_image_block_to_payload': True,
    'chunking_strategy': 'by_title',
    'max_characters': 10000,
    'combine_text_under_n_chars': 2000,
    'new_after_n_chars': 6000,
}

# LLM configuration
LLM_CONFIG = {
    'summarization_model': 'gpt-4o-mini',
    'image_description_model': 'gpt-4o-mini',
    'rag_response_model': 'gpt-4o-mini'
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    'default_k': 4,  # Number of documents to retrieve
    'search_type': 'similarity'
}

def ensure_directories():
    """Ensure all necessary directories exist."""
    directories = [
        SOP_DATA_DIR,
        SOP_MEMORY_DIR,
        CHROMA_PERSIST_PATH,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test configuration
    print("SOP RAG Configuration:")
    print(f"Base directory: {BASE_DIR}")
    print(f"SOP data directory: {SOP_DATA_DIR}")
    print(f"ChromaDB path: {CHROMA_PERSIST_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    
    # Ensure directories exist
    ensure_directories()
    print("\nDirectories created successfully!")