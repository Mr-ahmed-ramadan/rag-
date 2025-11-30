"""
Knowledge Extraction Backend
FastAPI server for document processing using IBM Docling
"""

import os
import logging
import asyncio
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
docling_ready = False

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


async def warm_up_docling():
    """Pre-initialize Docling in background to avoid timeout on first request"""
    global docling_ready
    try:
        logger.info("Warming up Docling...")
        proc = get_processor()
        docling_ready = proc.is_available()
        logger.info(f"Docling warm-up complete: available={docling_ready}")
    except Exception as e:
        logger.error(f"Docling warm-up failed: {e}")
        docling_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting Knowledge Extraction Backend v2.6.0...")
    logger.info(f"UPSTASH_SEARCH_REST_URL set: {bool(os.getenv('UPSTASH_SEARCH_REST_URL'))}")
    logger.info(f"UPSTASH_SEARCH_REST_TOKEN set: {bool(os.getenv('UPSTASH_SEARCH_REST_TOKEN'))}")
    
    asyncio.create_task(warm_up_docling())
    
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Knowledge Extraction Backend",
    description="Document processing API using IBM Docling",
    version="2.6.0",
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
    search_url = os.getenv('UPSTASH_SEARCH_REST_URL', '')
    search_token = os.getenv('UPSTASH_SEARCH_REST_TOKEN', '')
    vector_url = os.getenv('UPSTASH_VECTOR_REST_URL', '')
    vector_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN', '')
    
    vector_store_available = False
    vector_store_error = None
    vector_store_debug = {}
    
    try:
        static_check = VectorStore.check_available()
        vector_store_debug["static_check"] = static_check
        
        vs = get_vector_store()
        instance_check = vs.is_available()
        vector_store_debug["instance_check"] = instance_check
        
        vector_store_available = static_check and instance_check
    except Exception as e:
        vector_store_error = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Vector store check failed: {vector_store_error}")
    
    return {
        "status": "healthy",
        "code_version": "2.6.0",
        "services": {
            "docling": True,
            "docling_ready": docling_ready,
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
    """Debug endpoint to check environment variables"""
    search_url = os.getenv('UPSTASH_SEARCH_REST_URL', '')
    search_token = os.getenv('UPSTASH_SEARCH_REST_TOKEN', '')
    vector_url = os.getenv('UPSTASH_VECTOR_REST_URL', '')
    vector_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN', '')
    
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
        "code_version": "2.6.0",
        "docling_ready": docling_ready,
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
            "docling_ready": docling_ready,
            "vector_store_available": vs.is_available(),
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False, "error": str(e)}


@app.get("/test/docling")
async def test_docling():
    """Test if Docling can be loaded"""
    try:
        logger.info("Testing Docling load...")
        from processor import DocumentProcessor
        proc = DocumentProcessor()
        available = proc.is_available()
        logger.info(f"Docling loaded successfully, available: {available}")
        return {
            "success": True,
            "docling_available": available,
            "docling_ready": docling_ready,
            "message": "Docling loaded successfully"
        }
    except Exception as e:
        logger.error(f"Docling load failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "docling_available": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.post("/test/process-full")
async def test_process_full(request: ProcessRequest):
    """Test full processing pipeline with detailed logging"""
    steps = []
    
    try:
        # Step 1: Download
        steps.append({"step": "download_start", "status": "running"})
        logger.info(f"[TEST] Step 1: Downloading {request.blob_url}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(request.blob_url)
            response.raise_for_status()
            content = response.content
        
        steps[-1]["status"] = "success"
        steps[-1]["bytes"] = len(content)
        logger.info(f"[TEST] Step 1 complete: {len(content)} bytes")
        
        # Step 2: Get file extension
        steps.append({"step": "parse_extension", "status": "running"})
        file_ext = request.blob_url.split(".")[-1].lower().split("?")[0]
        steps[-1]["status"] = "success"
        steps[-1]["extension"] = file_ext
        logger.info(f"[TEST] Step 2 complete: extension={file_ext}")
        
        # Step 3: Initialize processor
        steps.append({"step": "init_processor", "status": "running"})
        logger.info("[TEST] Step 3: Initializing processor...")
        proc = get_processor()
        steps[-1]["status"] = "success"
        steps[-1]["available"] = proc.is_available()
        logger.info(f"[TEST] Step 3 complete: processor available={proc.is_available()}")
        
        # Step 4: Process with Docling
        steps.append({"step": "docling_process", "status": "running"})
        logger.info("[TEST] Step 4: Processing with Docling...")
        extracted = proc.process(content, file_ext, request.document_id)
        steps[-1]["status"] = "success"
        steps[-1]["content_length"] = len(extracted.get("content", ""))
        logger.info(f"[TEST] Step 4 complete: extracted {len(extracted.get('content', ''))} chars")
        
        # Step 5: Chunk
        steps.append({"step": "chunking", "status": "running"})
        logger.info("[TEST] Step 5: Chunking...")
        chunks = proc.chunk(extracted, request.document_id)
        steps[-1]["status"] = "success"
        steps[-1]["chunk_count"] = len(chunks)
        logger.info(f"[TEST] Step 5 complete: {len(chunks)} chunks")
        
        # Step 6: Store in vector DB
        steps.append({"step": "vector_store", "status": "running"})
        logger.info("[TEST] Step 6: Storing in vector DB...")
        vs = get_vector_store()
        
        ids = [chunk["id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        metadata_list = [chunk["metadata"] for chunk in chunks]
        
        vs.upsert(ids=ids, contents=contents, metadata=metadata_list)
        steps[-1]["status"] = "success"
        logger.info("[TEST] Step 6 complete: stored in vector DB")
        
        return {
            "success": True,
            "document_id": request.document_id,
            "chunk_count": len(chunks),
            "steps": steps,
            "message": "Full processing test passed!"
        }
        
    except Exception as e:
        logger.error(f"[TEST] Error: {str(e)}", exc_info=True)
        if steps and steps[-1]["status"] == "running":
            steps[-1]["status"] = "error"
            steps[-1]["error"] = str(e)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "steps": steps,
        }


@app.post("/test/process-simple")
async def test_process_simple(request: ProcessRequest):
    """Test processing without Docling - just download and store text"""
    try:
        logger.info(f"Simple test: downloading {request.blob_url}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(request.blob_url)
            response.raise_for_status()
            content = response.content
        
        logger.info(f"Downloaded {len(content)} bytes")
        
        text_content = f"Document {request.document_id}: {len(content)} bytes downloaded successfully"
        
        vs = get_vector_store()
        vs.upsert(
            ids=[f"{request.document_id}_test"],
            contents=[text_content],
            metadata=[{"document_id": request.document_id, "test": True}]
        )
        
        return {
            "success": True,
            "document_id": request.document_id,
            "bytes_downloaded": len(content),
            "message": "Simple test passed - download and vector store work"
        }
    except Exception as e:
        logger.error(f"Simple test failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def process_document_background(document_id: str, blob_url: str, filename: Optional[str] = None):
    """Process document in background - called asynchronously"""
    try:
        logger.info(f"[{document_id}] Background processing started")
        
        processing_status[document_id]["progress"] = 10
        
        # Download
        logger.info(f"[{document_id}] Downloading from {blob_url}")
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.get(blob_url)
            response.raise_for_status()
            content = response.content
        
        logger.info(f"[{document_id}] Downloaded {len(content)} bytes")
        processing_status[document_id]["progress"] = 20
        
        # Get file extension
        file_ext = blob_url.split(".")[-1].lower().split("?")[0]
        
        # Process with Docling (this is the slow part)
        logger.info(f"[{document_id}] Processing with Docling (ext: {file_ext})")
        processing_status[document_id]["progress"] = 30
        
        proc = get_processor()
        extracted = await asyncio.get_event_loop().run_in_executor(
            None, proc.process, content, file_ext, document_id
        )
        
        logger.info(f"[{document_id}] Docling processing complete")
        processing_status[document_id]["progress"] = 60
        
        # Chunk
        logger.info(f"[{document_id}] Chunking document")
        chunks = proc.chunk(extracted, document_id)
        logger.info(f"[{document_id}] Created {len(chunks)} chunks")
        processing_status[document_id]["progress"] = 70
        
        # Store in vector DB
        logger.info(f"[{document_id}] Storing in Upstash Search")
        vs = get_vector_store()
        
        final_filename = filename or extracted["metadata"].get("title", "Unknown")
        
        ids = [chunk["id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        metadata_list = []
        
        for chunk in chunks:
            meta = chunk["metadata"].copy()
            meta["filename"] = final_filename
            meta["document_id"] = document_id
            metadata_list.append(meta)
        
        vs.upsert(ids=ids, contents=contents, metadata=metadata_list)
        
        processing_status[document_id]["progress"] = 90
        
        # Mark as ready
        processing_status[document_id] = {
            "status": "ready",
            "progress": 100,
            "chunk_count": len(chunks),
            "error": None,
        }
        
        logger.info(f"[{document_id}] Processing complete: {len(chunks)} chunks stored")
        
    except Exception as e:
        logger.error(f"[{document_id}] Background processing error: {str(e)}", exc_info=True)
        processing_status[document_id] = {
            "status": "error",
            "progress": 0,
            "chunk_count": None,
            "error": str(e),
        }


@app.post("/process")
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Start document processing in background - returns immediately"""
    document_id = request.document_id
    
    # Initialize status
    processing_status[document_id] = {
        "status": "processing",
        "progress": 0,
        "chunk_count": None,
        "error": None,
    }
    
    # Add to background tasks - this returns immediately
    background_tasks.add_task(
        process_document_background,
        document_id,
        request.blob_url,
        request.filename
    )
    
    logger.info(f"[{document_id}] Processing started in background")
    
    # Return immediately with "accepted" status
    return {
        "success": True,
        "document_id": document_id,
        "status": "processing",
        "message": "Document processing started. Poll /status/{document_id} for progress.",
    }


@app.get("/status/{document_id}", response_model=StatusResponse)
async def get_status(document_id: str):
    """Get processing status for a document"""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    status = processing_status[document_id]
    return StatusResponse(document_id=document_id, **status)


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks"""
    try:
        vs = get_vector_store()
        results = vs.query(query_text=request.query, top_k=request.top_k)
        
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
        results = vs.query(query_text="", top_k=1, filter_={"id": chunk_id})
        
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
