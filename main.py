"""
Knowledge Extraction Backend
FastAPI server for document processing using IBM Docling
"""

import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

from processor import DocumentProcessor
from embeddings import EmbeddingService
from vector_store import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()

# Processing status tracker
processing_status = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting Knowledge Extraction Backend...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Knowledge Extraction Backend",
    description="Document processing API using IBM Docling",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    document_id: str
    blob_url: str
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None


class ProcessResponse(BaseModel):
    success: bool
    document_id: str
    chunk_count: int
    message: str


class StatusResponse(BaseModel):
    document_id: str
    status: str
    progress: Optional[int] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]


class ChunkRequest(BaseModel):
    excerpt: str
    page_number: Optional[int] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "docling": processor.is_available(),
            "embeddings": embedding_service.is_available(),
            "vector_store": vector_store.is_available(),
        }
    }


@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a document and store embeddings
    """
    document_id = request.document_id
    
    # Initialize status
    processing_status[document_id] = {
        "status": "processing",
        "progress": 0,
        "chunk_count": None,
        "error": None,
    }
    
    try:
        # Download document
        logger.info(f"Downloading document {document_id} from {request.blob_url}")
        processing_status[document_id]["progress"] = 10
        
        async with httpx.AsyncClient() as client:
            response = await client.get(request.blob_url)
            response.raise_for_status()
            content = response.content
        
        # Get file extension from URL
        file_ext = request.blob_url.split(".")[-1].lower().split("?")[0]
        
        # Process with Docling
        logger.info(f"Processing document {document_id} with Docling")
        processing_status[document_id]["progress"] = 30
        
        extracted = processor.process(content, file_ext, document_id)
        
        # Chunk content
        logger.info(f"Chunking document {document_id}")
        processing_status[document_id]["progress"] = 50
        
        chunks = processor.chunk(extracted, document_id)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        processing_status[document_id]["progress"] = 70
        
        provider = request.embedding_provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER", "local")
        model = request.embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        embeddings = embedding_service.embed(
            [chunk["content"] for chunk in chunks],
            provider=provider,
            model=model,
        )
        
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["content"] = chunk["content"]
            chunk["metadata"]["filename"] = extracted["metadata"].get("title", "Unknown")
        
        # Store in vector database
        logger.info(f"Storing {len(chunks)} vectors in Upstash")
        processing_status[document_id]["progress"] = 90
        
        vector_store.upsert(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=embeddings,
            metadata=[chunk["metadata"] for chunk in chunks],
        )
        
        # Update status
        processing_status[document_id] = {
            "status": "ready",
            "progress": 100,
            "chunk_count": len(chunks),
            "error": None,
        }
        
        logger.info(f"Document {document_id} processed successfully: {len(chunks)} chunks")
        
        return ProcessResponse(
            success=True,
            document_id=document_id,
            chunk_count=len(chunks),
            message="Document processed successfully",
        )
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        processing_status[document_id] = {
            "status": "error",
            "progress": 0,
            "chunk_count": None,
            "error": str(e),
        }
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{document_id}", response_model=StatusResponse)
async def get_status(document_id: str):
    """Get processing status for a document"""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    status = processing_status[document_id]
    return StatusResponse(
        document_id=document_id,
        **status,
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant document chunks
    """
    try:
        # Generate embedding for query
        provider = request.embedding_provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER", "local")
        model = request.embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        query_embedding = embedding_service.embed(
            [request.query],
            provider=provider,
            model=model,
        )[0]
        
        # Search vector store
        results = vector_store.query(
            embedding=query_embedding,
            top_k=request.top_k,
        )
        
        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    metadata=r["metadata"],
                )
                for r in results
            ]
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chunk/{document_id}")
async def get_chunk(document_id: str, request: ChunkRequest):
    """
    Get full chunk content for a citation
    """
    try:
        # Search for the chunk by excerpt
        provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "local")
        model = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        query_embedding = embedding_service.embed(
            [request.excerpt],
            provider=provider,
            model=model,
        )[0]
        
        # Search with filter for this document
        results = vector_store.query(
            embedding=query_embedding,
            top_k=1,
            filter_={"document_id": document_id},
        )
        
        if results:
            return {"content": results[0]["metadata"].get("content", request.excerpt)}
        
        return {"content": request.excerpt}
    except Exception as e:
        logger.error(f"Chunk retrieval error: {str(e)}")
        return {"content": request.excerpt}


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    try:
        # Delete vectors by prefix
        vector_store.delete_by_prefix(document_id)
        
        # Remove from status tracker
        if document_id in processing_status:
            del processing_status[document_id]
        
        return {"success": True, "message": f"Document {document_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
