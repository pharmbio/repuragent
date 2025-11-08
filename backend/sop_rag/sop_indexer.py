import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

import sys
from pathlib import Path

# Add current directory to path to find config when run from root
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import (
    SOP_DATA_DIR, CHROMA_PERSIST_PATH, COLLECTION_NAME, ID_KEY, DOCSTORE_PATH,
    PDF_PROCESSING_CONFIG, LLM_CONFIG, ensure_directories
)

load_dotenv()

def discover_pdf_files(directory: str) -> List[str]:
    """Discover all PDF files in the specified directory."""
    pdf_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    for file_path in directory_path.glob("*.pdf"):
        if file_path.is_file():
            pdf_files.append(str(file_path))
    
    return sorted(pdf_files)

def extract_pdf_content(file_path: str) -> List[Any]:
    """Extract and chunk PDF content including text, tables, and images."""
    return partition_pdf(
        filename=file_path,
        **PDF_PROCESSING_CONFIG
    )

def separate_content_types(chunks: List[Any]) -> Dict[str, List[Any]]:
    """Separate chunks into tables, texts, and images."""
    tables = []
    texts = []
    
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    
    return {"tables": tables, "texts": texts}

def extract_images_base64(chunks: List[Any]) -> List[str]:
    """Extract base64-encoded images from CompositeElement chunks."""
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if 'Image' in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def create_text_table_summarizer() -> RunnableLambda:
    """Create a chain for summarizing text and table content."""
    prompt_text_tables = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""
    template = ChatPromptTemplate.from_template(prompt_text_tables)
    llm = ChatOpenAI(model=LLM_CONFIG['summarization_model'])
    return template | llm | StrOutputParser()

def create_image_summarizer() -> RunnableLambda:
    """Create a chain for summarizing image content."""
    prompt_images = """Describe the image in detail. For context,
the image is part of a research paper explaining the transformers
architecture. Be specific about graphs, such as bar plots."""
    
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_images},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt | ChatOpenAI(model=LLM_CONFIG['image_description_model']) | StrOutputParser()

def create_multi_vector_retriever():
    """Create MultiVectorRetriever with persistent ChromaDB and LocalFileStore."""
    # Ensure directory exists
    ensure_directories()
    
    # Create vector store for summaries (search)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(CHROMA_PERSIST_PATH)
    )
    
    # Create docstore for original content (retrieval)
    docstore = LocalFileStore(str(DOCSTORE_PATH.parent / "docstore"))
    
    # Create MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY
    )
    
    return retriever

def add_content_to_retriever(retriever: MultiVectorRetriever, content: List[Any], summaries: List[str]) -> None:
    """Add content and summaries to the MultiVectorRetriever."""
    doc_ids = [str(uuid.uuid4()) for _ in content]
    summary_docs = []
    original_docs = []
    
    for i, summary in enumerate(summaries):
        # Create summary document for vector store (used for search)
        summary_metadata = {
            ID_KEY: doc_ids[i],
            'content_type': str(type(content[i]).__name__)
        }
        
        # Preserve filename if available
        if hasattr(content[i], 'metadata') and hasattr(content[i].metadata, 'filename'):
            summary_metadata['filename'] = content[i].metadata.filename
            
        summary_docs.append(Document(page_content=summary, metadata=summary_metadata))
        
        # Create original document for docstore (retrieved content)
        original_text = ""
        if hasattr(content[i], 'text'):
            original_text = content[i].text
        elif hasattr(content[i], 'metadata') and hasattr(content[i].metadata, 'text_as_html'):
            original_text = content[i].metadata.text_as_html
        elif isinstance(content[i], str):  # For images
            original_text = f"[Base64 Image Data: {len(content[i])} characters]"
            
        original_docs.append(Document(page_content=original_text, metadata=summary_metadata))
    
    # Add documents to the retriever
    retriever.vectorstore.add_documents(summary_docs)
    # Store original documents with metadata as JSON in docstore
    import json
    doc_data = []
    for doc in original_docs:
        doc_dict = {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        doc_data.append(json.dumps(doc_dict).encode('utf-8'))
    
    retriever.docstore.mset(list(zip(doc_ids, doc_data)))

def clear_existing_collection():
    """Clear existing collection and docstore to rebuild from scratch."""
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(CHROMA_PERSIST_PATH)
        )
        vectorstore.delete_collection()
        print("Cleared existing collection")
    except Exception as e:
        print(f"No existing collection to clear: {e}")
    
    # Clear docstore
    try:
        docstore_dir = DOCSTORE_PATH.parent / "docstore"
        if docstore_dir.exists():
            import shutil
            shutil.rmtree(docstore_dir)
            print("Cleared existing docstore")
    except Exception as e:
        print(f"No existing docstore to clear: {e}")

def process_and_index_pdfs(directory: str) -> None:
    """Process all PDF files and store in persistent vector database."""    
    # Clear existing collection
    clear_existing_collection()
    
    # Discover PDF files
    pdf_files = discover_pdf_files(str(SOP_DATA_DIR))
    print(f"Found {len(pdf_files)} PDF files to process:")
    for file in pdf_files:
        print(f"  - {os.path.basename(file)}")
    
    if not pdf_files:
        print("No PDF files found to process")
        return
    
    # Create MultiVectorRetriever
    retriever = create_multi_vector_retriever()
    
    # Create summarizers
    text_table_summarizer = create_text_table_summarizer()
    image_summarizer = create_image_summarizer()
    
    total_texts = 0
    total_tables = 0
    total_images = 0
    
    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\nProcessing {os.path.basename(pdf_file)}...")
        
        try:
            # Extract content from current PDF
            chunks = extract_pdf_content(pdf_file)
            content_types = separate_content_types(chunks)
            texts = content_types["texts"]
            tables = content_types["tables"]
            images = extract_images_base64(chunks)
            
            # Process texts
            if texts:
                print(f"  - Processing {len(texts)} text chunks...")
                text_summaries = text_table_summarizer.batch(texts)
                add_content_to_retriever(retriever, texts, text_summaries)
                total_texts += len(texts)
            
            # Process tables
            if tables:
                print(f"  - Processing {len(tables)} tables...")
                tables_html = [table.metadata.text_as_html for table in tables]
                table_summaries = text_table_summarizer.batch(tables_html)
                add_content_to_retriever(retriever, tables, table_summaries)
                total_tables += len(tables)
            
            # Process images
            if images:
                print(f"  - Processing {len(images)} images...")
                image_summaries = image_summarizer.batch(images)
                add_content_to_retriever(retriever, images, image_summaries)
                total_images += len(images)
                
        except Exception as e:
            print(f"  - Error processing {pdf_file}: {e}")
            continue
    
    print(f"\n=== Indexing Complete ===")
    print(f"Total indexed: {total_texts} text chunks, {total_tables} tables, {total_images} images")
    print(f"Vector store saved to: {CHROMA_PERSIST_PATH}")

def main():
    """Main execution function."""
    process_and_index_pdfs(str(SOP_DATA_DIR))

if __name__ == "__main__":
    main()