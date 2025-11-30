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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from vector_store import VectorStore

processor = None
vector_store = None

def get_processor():
    """Lazy-load the document processor"""
    global processor
    if processor is None:
        from processor import DocumentProcessor
        processor = DocumentProcessor()
    return processor

def get_vector_store():
    """Lazy-load the vector store"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store

# Processing status tracker
processing_status = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting Knowledge Extraction Backend...")
    logger.info(f"UPSTASH_SEARCH_REST_URL set: {bool(os.getenv('UPSTASH_SEARCH_REST_URL'))}")
    logger.info(f"UPSTASH_SEARCH_REST_TOKEN set: {bool(os.getenv('UPSTASH_SEARCH_REST_TOKEN'))}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Knowledge Extraction Backend",
    description="Document processing API using IBM Docling",
    version="2.3.0",  # Updated version
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    document_id: str
    blob_url: str
    filename: Optional[str] = None


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


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.get("/health")
async def health_check():
    """Health check endpoint with full diagnostics"""
    # Check environment variables
    search_url = os.getenv('UPSTASH_SEARCH_REST_URL', '')
    search_token = os.getenv('UPSTASH_SEARCH_REST_TOKEN', '')
    vector_url = os.getenv('UPSTASH_VECTOR_REST_URL', '')
    vector_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN', '')
    
    upstash_url_set = bool(search_url or vector_url)
    upstash_token_set = bool(search_token or vector_token)
    
    # Test vector store availability
    vector_store_available = False
    vector_store_error = None
    vector_store_debug = {}
    
    try:
        static_check = VectorStore.check_available()
        vector_store_debug["static_check"] = static_check
        
        # Also try instance method
        vs = get_vector_store()
        instance_check = vs.is_available()
        vector_store_debug["instance_check"] = instance_check
        
        vector_store_available = static_check and instance_check
    except Exception as e:
        vector_store_error = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Vector store check failed: {vector_store_error}")
    
    return {
        "status": "healthy",
        "code_version": "2.3.0",  # Updated version
        "services": {
            "docling": True,
            "vector_store": vector_store_available,
        },
        "environment": {
            "UPSTASH_SEARCH_REST_URL": bool(search_url),
            "UPSTASH_SEARCH_REST_TOKEN": bool(search_token),
            "UPSTASH_VECTOR_REST_URL": bool(vector_url),
            "UPSTASH_VECTOR_REST_TOKEN": bool(vector_token),
            "url_length": len(search_url or vector_url),
            "token_length": len(search_token or vector_token),
        },
        "debug": {
            "vector_store_error": vector_store_error,
            "vector_store_debug": vector_store_debug,
        }
    }


@app.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables (redacted for security)"""
    search_url = os.getenv('UPSTASH_SEARCH_REST_URL', '')
    search_token = os.getenv('UPSTASH_SEARCH_REST_TOKEN', '')
    vector_url = os.getenv('UPSTASH_VECTOR_REST_URL', '')
    vector_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN', '')
    
    # Get all env var names (not values for security)
    all_env_keys = sorted([k for k in os.environ.keys() if 'UPSTASH' in k.upper()])
    
    return {
        "upstash_search_url": {
            "set": bool(search_url),
            "length": len(search_url),
            "prefix": search_url[:30] + "..." if len(search_url) > 30 else search_url,
        },
        "upstash_search_token": {
            "set": bool(search_token),
            "length": len(search_token),
        },
        "upstash_vector_url": {
            "set": bool(vector_url),
            "length": len(vector_url),
        },
        "upstash_vector_token": {
            "set": bool(vector_token),
            "length": len(vector_token),
        },
        "all_upstash_env_keys": all_env_keys,
        "code_version": "2.3.0",  # Updated version
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - verifies all services are loaded"""
    try:
        proc = get_processor()
        vs = get_vector_store()
        return {
            "ready": True,
            "docling_available": proc.is_available(),
            "vector_store_available": vs.is_available(),
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False, "error": str(e)}


@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a document and store in vector database
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
        proc = get_processor()
        vs = get_vector_store()
        
        # Download document
        logger.info(f"Downloading document {document_id} from {request.blob_url}")
        processing_status[document_id]["progress"] = 10
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(request.blob_url)
            response.raise_for_status()
            content = response.content
        
        logger.info(f"Downloaded {len(content)} bytes")
        
        # Get file extension from URL
        file_ext = request.blob_url.split(".")[-1].lower().split("?")[0]
        
        # Process with Docling
        logger.info(f"Processing document {document_id} with Docling (ext: {file_ext})")
        processing_status[document_id]["progress"] = 30
        
        extracted = proc.process(content, file_ext, document_id)
        
        # Chunk content
        logger.info(f"Chunking document {document_id}")
        processing_status[document_id]["progress"] = 50
        
        chunks = proc.chunk(extracted, document_id)
        
        logger.info(f"Storing {len(chunks)} chunks in Upstash Search")
        processing_status[document_id]["progress"] = 70
        
        # Get filename from request or extracted metadata
        filename = request.filename or extracted["metadata"].get("title", "Unknown")
        
        # Prepare data for vector store
        ids = [chunk["id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        metadata_list = []
        
        for chunk in chunks:
            meta = chunk["metadata"].copy()
            meta["filename"] = filename
            meta["document_id"] = document_id
            metadata_list.append(meta)
        
        # Store in vector database
        vs.upsert(
            ids=ids,
            contents=contents,
            metadata=metadata_list,
        )
        
        processing_status[document_id]["progress"] = 90
        
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
        logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
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
    """Search for relevant document chunks"""
    try:
        vs = get_vector_store()
        results = vs.query(
            query_text=request.query,
            top_k=request.top_k,
        )
        
        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    content=r.get("data") or r.get("metadata", {}).get("content", ""),
                    metadata=r["metadata"],
                )
                for r in results
            ]
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunk/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Get full chunk content for a citation"""
    try:
        vs = get_vector_store()
        results = vs.query(
            query_text="",
            top_k=1,
            filter_={"id": chunk_id},
        )
        
        if results:
            return {
                "content": results[0].get("data") or results[0].get("metadata", {}).get("content", ""),
                "metadata": results[0].get("metadata", {}),
            }
        
        return {"content": "", "metadata": {}}
    except Exception as e:
        logger.error(f"Chunk retrieval error: {str(e)}")
        return {"content": "", "metadata": {}}


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    try:
        vs = get_vector_store()
        vs.delete_by_prefix(document_id)
        
        if document_id in processing_status:
            del processing_status[document_id]
        
        return {"success": True, "message": f"Document {document_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
