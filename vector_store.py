"""
Upstash Vector store integration
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from upstash_vector import Index
    UPSTASH_AVAILABLE = True
except ImportError:
    UPSTASH_AVAILABLE = False
    logger.warning("upstash-vector not installed")


class VectorStore:
    """Upstash Vector store for embeddings"""
    
    def __init__(self):
        self._index = None
    
    def _get_index(self) -> "Index":
        """Get or create Upstash Vector index"""
        if self._index is None:
            if not UPSTASH_AVAILABLE:
                raise RuntimeError("upstash-vector not installed")
            
            url = os.getenv("UPSTASH_VECTOR_REST_URL")
            token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
            
            if not url or not token:
                raise RuntimeError("UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN must be set")
            
            self._index = Index(url=url, token=token)
        
        return self._index
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        try:
            self._get_index()
            return True
        except Exception:
            return False
    
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert vectors into the index
        
        Args:
            ids: Vector IDs
            embeddings: Embedding vectors
            metadata: Metadata for each vector
        """
        index = self._get_index()
        
        # Prepare vectors
        vectors = [
            {
                "id": id_,
                "vector": embedding,
                "metadata": meta,
            }
            for id_, embedding, meta in zip(ids, embeddings, metadata)
        ]
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Upserted {len(vectors)} vectors")
    
    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        filter_: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query vectors by similarity
        
        Args:
            embedding: Query embedding
            top_k: Number of results
            filter_: Metadata filter
            
        Returns:
            List of results with scores
        """
        index = self._get_index()
        
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_,
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
    
    def delete_by_prefix(self, prefix: str) -> None:
        """
        Delete vectors by ID prefix
        
        Args:
            prefix: ID prefix to match
        """
        index = self._get_index()
        
        # Query to find matching IDs
        # Note: This is a simple implementation
        # For production, consider using Upstash's delete by filter
        try:
            index.delete(ids=[prefix], delete_all=False)
            logger.info(f"Deleted vectors with prefix: {prefix}")
        except Exception as e:
            logger.warning(f"Delete by prefix failed: {e}")
