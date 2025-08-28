"""
Embedding service for document processing and vector storage
"""
import asyncio
from typing import List, Optional
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, get_google_api_keys
from db.chroma_manager import chroma_manager
from services.file_service import file_service
from models.schemas import BuildStatus, BuildStatusEnum


class EmbeddingService:
    """Service for processing documents and creating embeddings"""
    
    def __init__(self):
        self.api_keys = get_google_api_keys()
        self.current_key_index = 0
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self._rag_chains = {}  # Company-specific RAG chains
        self.build_statuses = {}  # Company-specific build statuses
    
    def get_current_llm(self) -> ChatGoogleGenerativeAI:
        """Get LLM with current API key"""
        if not self.api_keys:
            raise ValueError("No Google API keys available")
        
        api_key = self.api_keys[self.current_key_index]
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key
        )
    
    def rotate_api_key(self):
        """Rotate to next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        chroma_manager.rotate_api_key()  # Also rotate chroma manager key
    
    async def process_company_documents(self, company_id: str) -> BuildStatus:
        """Process all documents for a company and build vector store"""
        
        # Initialize build status
        self.build_statuses[company_id] = BuildStatus(
            status=BuildStatusEnum.BUILDING,
            message="Starting document processing...",
            company_id=company_id,
            timestamp=datetime.now(),
            progress=0.0
        )
        
        try:
            # Load documents
            self.build_statuses[company_id].message = "Loading documents..."
            self.build_statuses[company_id].progress = 0.1
            
            documents = file_service.load_documents_from_company(company_id)
            
            if not documents:
                self.build_statuses[company_id] = BuildStatus(
                    status=BuildStatusEnum.ERROR,
                    message="No documents found for processing",
                    company_id=company_id,
                    timestamp=datetime.now()
                )
                return self.build_statuses[company_id]
            
            # Split documents
            self.build_statuses[company_id].message = f"Splitting {len(documents)} documents into chunks..."
            self.build_statuses[company_id].progress = 0.3
            
            splits = self.text_splitter.split_documents(documents)
            
            # Create/update vector store
            self.build_statuses[company_id].message = f"Creating embeddings for {len(splits)} chunks..."
            self.build_statuses[company_id].progress = 0.6
            
            # Check if collection exists
            existing_collections = chroma_manager.list_company_collections()
            
            if company_id in existing_collections:
                # Delete existing collection
                chroma_manager.delete_company_collection(company_id)
            
            # Create new collection
            success = chroma_manager.create_company_collection(company_id, splits)
            
            if not success:
                self.build_statuses[company_id] = BuildStatus(
                    status=BuildStatusEnum.ERROR,
                    message="Failed to create vector store",
                    company_id=company_id,
                    timestamp=datetime.now()
                )
                return self.build_statuses[company_id]
            
            # Initialize RAG chain
            self.build_statuses[company_id].message = "Initializing RAG pipeline..."
            self.build_statuses[company_id].progress = 0.9
            
            rag_chain = await self._create_rag_chain(company_id)
            if rag_chain:
                self._rag_chains[company_id] = rag_chain
            
            # Complete
            self.build_statuses[company_id] = BuildStatus(
                status=BuildStatusEnum.COMPLETED,
                message=f"Successfully processed {len(documents)} documents into {len(splits)} chunks",
                company_id=company_id,
                timestamp=datetime.now(),
                progress=1.0
            )
            
            return self.build_statuses[company_id]
            
        except Exception as e:
            self.build_statuses[company_id] = BuildStatus(
                status=BuildStatusEnum.ERROR,
                message=f"Processing failed: {str(e)}",
                company_id=company_id,
                timestamp=datetime.now()
            )
            
            # Try rotating API key for next attempt
            self.rotate_api_key()
            
            return self.build_statuses[company_id]
    
    async def _create_rag_chain(self, company_id: str):
        """Create RAG chain for a specific company"""
        try:
            llm = self.get_current_llm()
            vectorstore = chroma_manager.get_company_vectorstore(company_id)
            retriever = vectorstore.as_retriever()
            
            # Context prompt
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            # System prompt
            system_prompt = f"""You are an advanced AI assistant with expertise in understanding and explaining complex information for company {company_id}.
Your role is to answer user questions comprehensively using the provided knowledge base context.

Guidelines:
1. Always ground your answers in the provided context, but expand with reasoning, clarification, and related insights.
2. Provide clear, structured, and well-organized responses (use sections, bullet points, or lists where helpful).
3. Be detailed — explain concepts fully instead of giving short or vague replies.
4. Highlight key insights, important details, and actionable information.
5. If something is unclear in the context, infer the most likely explanation and explicitly state your assumptions.
6. If the information truly does not exist in the knowledge base, say: 
   "The available knowledge base does not provide a direct answer to this question," 
   and suggest possible directions or related knowledge.

Context:
{{context}}"""
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            return rag_chain
            
        except Exception as e:
            print(f"Error creating RAG chain for company {company_id}: {e}")
            self.rotate_api_key()
            return None
    
    async def query_company(self, company_id: str, query: str, chat_history: List[dict] = None) -> dict:
        """Query documents for a specific company"""
        
        # Check if RAG chain exists
        if company_id not in self._rag_chains:
            # Try to create RAG chain
            rag_chain = await self._create_rag_chain(company_id)
            if not rag_chain:
                raise ValueError(f"RAG system not available for company {company_id}")
            self._rag_chains[company_id] = rag_chain
        
        # Convert chat history to LangChain format
        chat_history_for_chain = []
        if chat_history:
            for msg in chat_history:
                content = msg.get('message', '')
                if msg.get('sender') == 'user':
                    chat_history_for_chain.append(HumanMessage(content=content))
                else:
                    chat_history_for_chain.append(AIMessage(content=content))
        
        try:
            # Query the RAG chain
            response = self._rag_chains[company_id].invoke({
                "input": query,
                "chat_history": chat_history_for_chain
            })
            
            # Extract sources
            sources = []
            if 'context' in response:
                for doc in response['context']:
                    if 'source' in doc.metadata:
                        sources.append(doc.metadata['source'])
            
            return {
                "response": response.get("answer", "I couldn't find an answer to that."),
                "sources": list(set(sources))  # Remove duplicates
            }
            
        except Exception as e:
            # Try rotating API key and retry once
            self.rotate_api_key()
            
            # Remove the failed RAG chain
            if company_id in self._rag_chains:
                del self._rag_chains[company_id]
            
            raise Exception(f"Query failed: {str(e)}")
    
    def get_build_status(self, company_id: str) -> BuildStatus:
        """Get build status for a company"""
        return self.build_statuses.get(company_id, BuildStatus(
            status=BuildStatusEnum.IDLE,
            message="No build process started",
            company_id=company_id,
            timestamp=datetime.now()
        ))
    
    def get_all_build_statuses(self) -> dict:
        """Get all build statuses"""
        return self.build_statuses
    
    async def add_document_to_company(self, company_id: str, file_id: str) -> bool:
        """Add a single new document to existing company vector store"""
        try:
            # Load the specific document
            file_info = file_service.get_file_info(company_id, file_id)
            if not file_info:
                return False
            
            # Load document content
            company_dir = file_service.get_company_directory(company_id)
            file_path = company_dir / file_info.filename
            
            if not file_path.exists():
                return False
            
            docs = file_service._load_document_by_type(file_path, file_info.extension)
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    'file_id': file_id,
                    'company_id': company_id,
                    'original_filename': file_info.original_filename,
                    'file_type': file_info.extension.replace('.', ''),
                    'created_at': file_info.created_at.isoformat()
                })
            
            # Split documents
            splits = self.text_splitter.split_documents(docs)
            
            # Add to vector store
            success = chroma_manager.add_documents_to_company(company_id, splits)
            
            if success and company_id in self._rag_chains:
                # Refresh RAG chain
                del self._rag_chains[company_id]
            
            return success
            
        except Exception as e:
            print(f"Error adding document {file_id} to company {company_id}: {e}")
            return False


# Global instance
embedding_service = EmbeddingService()
