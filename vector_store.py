"""
Upstash Search/Vector store integration
Uses Upstash's built-in embedding models - no external API key needed!
"""

import os
import logging
import httpx
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """Upstash Vector/Search store with built-in embeddings"""
    
    def __init__(self):
        self._url = None
        self._token = None
    
    def _get_credentials(self):
        """Get Upstash credentials - always check fresh from environment"""
        url = os.getenv("UPSTASH_SEARCH_REST_URL") or os.getenv("UPSTASH_VECTOR_REST_URL")
        token = os.getenv("UPSTASH_SEARCH_REST_TOKEN") or os.getenv("UPSTASH_VECTOR_REST_TOKEN")
        
        if not url or not token:
            raise RuntimeError(
                "UPSTASH_SEARCH_REST_URL/UPSTASH_SEARCH_REST_TOKEN or "
                "UPSTASH_VECTOR_REST_URL/UPSTASH_VECTOR_REST_TOKEN must be set"
            )
        
        self._url = url
        self._token = token
        return self._url, self._token
    
    @staticmethod
    def check_available() -> bool:
        """
        Static method to check if vector store is available without creating instance.
        Checks environment variables directly.
        """
        url = os.getenv("UPSTASH_SEARCH_REST_URL") or os.getenv("UPSTASH_VECTOR_REST_URL")
        token = os.getenv("UPSTASH_SEARCH_REST_TOKEN") or os.getenv("UPSTASH_VECTOR_REST_TOKEN")
        return bool(url and token)
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        try:
            self._get_credentials()
            return True
        except Exception as e:
            logger.warning(f"Vector store not available: {e}")
            return False
    
    def upsert(
        self,
        ids: List[str],
        contents: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert documents into the index using Upstash's built-in embeddings
        
        Args:
            ids: Document IDs
            contents: Text content (will be embedded by Upstash)
            metadata: Metadata for each document
        """
        url, token = self._get_credentials()
        
        # Prepare data for upsert-data endpoint (auto-embedding)
        data = [
            {
                "id": id_,
                "data": content,
                "metadata": {**meta, "content": content},  # Store content in metadata too
            }
            for id_, content, meta in zip(ids, contents, metadata)
        ]
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            response = httpx.post(
                f"{url}/upsert-data",
                headers={"Authorization": f"Bearer {token}"},
                json=batch,
                timeout=60.0,
            )
            
            if response.status_code != 200:
                logger.error(f"Upsert failed: {response.text}")
                raise RuntimeError(f"Upsert failed: {response.text}")
        
        logger.info(f"Upserted {len(data)} documents with auto-embedding")
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query documents by semantic similarity using Upstash's built-in embeddings
        
        Args:
            query_text: Query string (will be embedded by Upstash)
            top_k: Number of results
            filter_: Metadata filter
            
        Returns:
            List of results with scores
        """
        url, token = self._get_credentials()
        
        request_body = {
            "data": query_text,
            "topK": top_k,
            "includeMetadata": True,
            "includeData": True,
        }
        
        if filter_:
            request_body["filter"] = filter_
        
        response = httpx.post(
            f"{url}/query-data",
            headers={"Authorization": f"Bearer {token}"},
            json=request_body,
            timeout=30.0,
        )
        
        if response.status_code != 200:
            logger.error(f"Query failed: {response.text}")
            raise RuntimeError(f"Query failed: {response.text}")
        
        results = response.json()
        
        # Handle both response formats
        if isinstance(results, dict) and "result" in results:
            results = results["result"]
        
        return [
            {
                "id": r.get("id"),
                "score": r.get("score", 0),
                "metadata": r.get("metadata", {}),
                "data": r.get("data", ""),
            }
            for r in results
        ]
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
        """
        url, token = self._get_credentials()
        
        response = httpx.post(
            f"{url}/delete",
            headers={"Authorization": f"Bearer {token}"},
            json=ids,
            timeout=30.0,
        )
        
        if response.status_code != 200:
            logger.warning(f"Delete failed: {response.text}")
        else:
            logger.info(f"Deleted {len(ids)} documents")
    
    def delete_by_prefix(self, prefix: str) -> None:
        """
        Delete documents by ID prefix
        Note: This queries first to find matching IDs, then deletes
        
        Args:
            prefix: ID prefix to match
        """
        url, token = self._get_credentials()
        
        try:
            # Use range query to find IDs with prefix
            response = httpx.post(
                f"{url}/range",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "cursor": "0",
                    "limit": 1000,
                    "prefix": prefix,
                },
                timeout=30.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                if "vectors" in data:
                    ids_to_delete = [v["id"] for v in data["vectors"]]
                    if ids_to_delete:
                        self.delete_by_ids(ids_to_delete)
                        return
            
            # Fallback: try direct delete
            self.delete_by_ids([prefix])
            
        except Exception as e:
            logger.warning(f"Delete by prefix failed: {e}")
