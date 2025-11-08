import os
from base64 import b64decode
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

import sys
from pathlib import Path

# Add current directory to path to find config when run from root
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import (
    CHROMA_PERSIST_PATH, COLLECTION_NAME, LLM_CONFIG, RETRIEVAL_CONFIG, ID_KEY, DOCSTORE_PATH
)

load_dotenv()

class SOPRetriever:
    def __init__(self):
        self.retriever = None
        self.rag_chain = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the MultiVectorRetriever and RAG chain."""
        if not os.path.exists(str(CHROMA_PERSIST_PATH)):
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_PERSIST_PATH}. "
                "Please run sop_indexer.py first to create the index."
            )
        
        docstore_dir = DOCSTORE_PATH.parent / "docstore"
        if not os.path.exists(str(docstore_dir)):
            raise FileNotFoundError(
                f"Document store not found at {docstore_dir}. "
                "Please run sop_indexer.py first to create the index."
            )
        
        # Create vector store
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(CHROMA_PERSIST_PATH)
        )
        
        # Check if collection has documents
        if vectorstore._collection.count() == 0:
            raise ValueError(
                "Vector store is empty. Please run sop_indexer.py to index documents."
            )
        
        print(f"Loaded vector store with {vectorstore._collection.count()} documents")
        
        # Create docstore
        docstore = LocalFileStore(str(docstore_dir))
        
        # Create MultiVectorRetriever
        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=ID_KEY,
            search_kwargs={"k": RETRIEVAL_CONFIG['default_k']}
        )
        
        print(f"MultiVectorRetriever created with vectorstore and docstore")
        print(f"Docstore contains {len(list(docstore.yield_keys()))} documents")
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain(self.retriever)
    
    def _create_rag_chain(self, retriever):
        """Create the complete RAG chain for question answering."""
        return {
            "context": retriever | RunnableLambda(self._parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self._build_prompt)
                | ChatOpenAI(model=LLM_CONFIG['rag_response_model'])
                | StrOutputParser()
            )
        )
    
    def _convert_bytes_to_docs(self, retrieved_items: List[Any]) -> List[Any]:
        """Convert bytes from docstore back to Document objects."""
        documents = []
        for item in retrieved_items:
            if isinstance(item, bytes):
                try:
                    # Try to deserialize JSON data with metadata
                    import json
                    from langchain.schema.document import Document
                    doc_dict = json.loads(item.decode('utf-8'))
                    documents.append(Document(
                        page_content=doc_dict['page_content'],
                        metadata=doc_dict['metadata']
                    ))
                except (json.JSONDecodeError, KeyError):
                    # Fallback: treat as plain text
                    from langchain.schema.document import Document
                    content = item.decode('utf-8')
                    documents.append(Document(page_content=content))
            elif hasattr(item, 'page_content'):
                # Already a Document
                documents.append(item)
            else:
                # Handle other cases
                from langchain.schema.document import Document
                documents.append(Document(page_content=str(item)))
        return documents
    
    def _parse_docs(self, docs: List[Any]) -> Dict[str, List[Any]]:
        """Split base64-encoded images and texts."""
        # First convert bytes to documents if needed
        docs = self._convert_bytes_to_docs(docs)
        
        b64 = []
        text = []
        for doc in docs:
            try:
                # Check if it's a base64 image
                b64decode(doc.page_content)
                b64.append(doc.page_content)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}
    
    def _build_prompt(self, kwargs: Dict[str, Any]) -> ChatPromptTemplate:
        """Build a prompt with context and question for RAG queries."""
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.page_content + "\n\n"

        prompt_template = f"""
Answer the question based only on the following context, which can include text, tables, and the below image.
Context: {context_text}
Question: {user_question}
"""

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                })

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        if self.rag_chain is None:
            raise ValueError("RAG chain not initialized")
        
        response = self.rag_chain.invoke(question)
        return response
    
    def get_sources(self, question: str) -> List[str]:
        """Get source documents for a question without generating response."""
        
        # First check what's in vectorstore (summaries)
        summary_docs = self.retriever.vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_CONFIG['default_k']}
        ).invoke(question)
        
        # Then get full documents from MultiVectorRetriever (docstore)
        docs = self.retriever.invoke(question)
        
        docs = self._convert_bytes_to_docs(docs)
        
        sources = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'filename' in doc.metadata:
                sources.append(os.path.basename(doc.metadata['filename']))
        
        return list(set(sources))  # Remove duplicates

def query_sop(question: str) -> Dict[str, Any]:
    """Convenience function for querying SOPs."""
    retriever = SOPRetriever()
    return retriever.query(question)

def main(query):
    """Main execution function for testing."""
    try:
        retriever = SOPRetriever()
        
        print(f"\nQuery: {query}")
        print("-" * 50)
            
        try:
            response = retriever.query(query)
            print(f"Response: {response['response']}")
                
            # Show sources with text content
            sources = retriever.get_sources(query)
            if sources:
                print(f"\nSources:")
                for source in sorted(sources):
                    print(f"  - {source}")
                
                
                # Get and display the actual retrieved documents (full content from docstore)
                docs = retriever.retriever.invoke(query)
                docs = retriever._convert_bytes_to_docs(docs)
                
                print(f"\nRetrieved Content:")
                for i, doc in enumerate(docs, 1):
                    filename = os.path.basename(doc.metadata.get('filename', 'Unknown')) if hasattr(doc, 'metadata') else 'Unknown'
                    print(f"\n--- Document {i} ---")
                    print(f"Source: {filename}")
                    print(f"Original Text: {doc.page_content}")  
                
        except Exception as e:
            print(f"Error processing query: {e}")
            
        print("\n" + "="*60)
    
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        print("Please run sop_indexer.py first to create the index.")

if __name__ == "__main__":
    query = input('Query:')
    main(query)