# build_db.py - One-time script to build the vector store

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Define the path for the persistent database
CHROMA_DB_PATH = "./chroma_db"

def load_documents_from_directory(directory_path):
    """
    Load documents from directory supporting multiple file formats:
    - .txt files
    - .pdf files  
    - .docx files
    - .json files
    """
    print(f"Loading documents from: {directory_path}")
    all_documents = []
    
    # Get all supported files
    directory = Path(directory_path)
    
    # Define supported file patterns
    supported_files = {
        '*.txt': TextLoader,
        '*.pdf': PyPDFLoader,
        '*.docx': Docx2txtLoader,
        '*.json': None  # We'll handle JSON separately
    }
    
    # Count files by type
    file_counts = {}
    
    for pattern, loader_class in supported_files.items():
        files = list(directory.glob(pattern))
        file_counts[pattern] = len(files)
        
        if pattern == '*.json':
            # Handle JSON files separately
            for file_path in files:
                try:
                    print(f"Loading JSON file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Convert JSON to text format
                    if isinstance(json_data, dict):
                        text_content = json.dumps(json_data, indent=2, ensure_ascii=False)
                    elif isinstance(json_data, list):
                        text_content = '\n'.join([
                            json.dumps(item, indent=2, ensure_ascii=False) 
                            if isinstance(item, (dict, list)) 
                            else str(item) 
                            for item in json_data
                        ])
                    else:
                        text_content = str(json_data)
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            'source': str(file_path),
                            'file_type': 'json',
                            'filename': file_path.name
                        }
                    )
                    all_documents.append(doc)
                    
                except Exception as e:
                    print(f"ERROR loading JSON file {file_path}: {e}")
                    continue
        else:
            # Handle other file types with their respective loaders
            for file_path in files:
                try:
                    print(f"Loading {pattern} file: {file_path}")
                    
                    if loader_class == TextLoader:
                        loader = loader_class(str(file_path), encoding='utf-8')
                    else:
                        loader = loader_class(str(file_path))
                    
                    docs = loader.load()
                    
                    # Add file type to metadata
                    for doc in docs:
                        doc.metadata['file_type'] = pattern.replace('*.', '')
                        doc.metadata['filename'] = file_path.name
                    
                    all_documents.extend(docs)
                    
                except Exception as e:
                    print(f"ERROR loading file {file_path}: {e}")
                    continue
    
    # Print summary
    print(f"\n📊 File Summary:")
    for pattern, count in file_counts.items():
        if count > 0:
            print(f"  - {pattern}: {count} files")
    
    print(f"\n✅ Total documents loaded: {len(all_documents)}")
    return all_documents

def build_vector_store():
    """
    Builds the vector store from the knowledge base and saves it to disk.
    Supports PDF, TXT, DOCX, and JSON files.
    """
    print("--- Starting Enhanced Vector Store Build Process ---")
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY_1")
    if not google_api_key:
        print("CRITICAL ERROR: GOOGLE_API_KEY_1 not found in environment. Build cannot proceed.")
        exit(1)
    else:
        print("SUCCESS: GOOGLE_API_KEY loaded successfully.")

    # 1. Load documents with multi-format support
    print("Step 1: Loading documents from './knowledge_base/' (PDF, TXT, DOCX, JSON)...")
    
    # Check if knowledge_base directory exists
    if not os.path.exists('./knowledge_base/'):
        print("CRITICAL ERROR: 'knowledge_base' directory not found. Build cannot proceed.")
        exit(1)
    
    docs = load_documents_from_directory('./knowledge_base/')
    
    if not docs:
        print("CRITICAL ERROR: No supported documents found in 'knowledge_base'. Build failed.")
        print("Supported formats: .txt, .pdf, .docx, .json")
        exit(1)

    # 2. Split documents
    print("\nStep 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"SUCCESS: Split documents into {len(splits)} chunks.")

    # 3. Create embeddings model
    print("\nStep 3: Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    print("SUCCESS: Embeddings model initialized.")

    # 4. Remove existing database if it exists
    if os.path.exists(CHROMA_DB_PATH):
        print(f"⚠️  Removing existing database at: {CHROMA_DB_PATH}")
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)

    # 5. Create Chroma vector store and PERSIST it to disk
    print(f"\nStep 4: Creating and persisting vector store at: {CHROMA_DB_PATH}")
    try:
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        print("SUCCESS: Vector store created and persisted.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create Chroma database. Error: {e}")
        exit(1)
    
    # 6. Verification Step
    print("\nStep 5: Verifying database creation...")
    if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
        print("--- ✅ SUCCESS: Enhanced Vector store built and verified successfully! ---")
        print(f"📁 Database location: {CHROMA_DB_PATH}")
        print(f"📚 Total chunks: {len(splits)}")
        print(f"📄 Source documents: {len(docs)}")
    else:
        print("--- CRITICAL ERROR: Vector store directory is empty or was not created. Build failed. ---")
        exit(1)

if __name__ == '__main__':
    build_vector_store()
