"""
Multi-provider embedding service
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try to import embedding providers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class EmbeddingService:
    """Multi-provider embedding service"""
    
    def __init__(self):
        self._local_models = {}
        self._openai_client = None
        self._cohere_client = None
    
    def is_available(self) -> bool:
        """Check if any embedding provider is available"""
        return (
            SENTENCE_TRANSFORMERS_AVAILABLE or
            (OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")) or
            (COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"))
        )
    
    def _get_local_model(self, model_name: str) -> "SentenceTransformer":
        """Get or load a local sentence transformer model"""
        if model_name not in self._local_models:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError("sentence-transformers not installed")
            self._local_models[model_name] = SentenceTransformer(model_name)
        return self._local_models[model_name]
    
    def _get_openai_client(self):
        """Get OpenAI client"""
        if self._openai_client is None:
            if not OPENAI_AVAILABLE:
                raise RuntimeError("openai package not installed")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client
    
    def _get_cohere_client(self):
        """Get Cohere client"""
        if self._cohere_client is None:
            if not COHERE_AVAILABLE:
                raise RuntimeError("cohere package not installed")
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise RuntimeError("COHERE_API_KEY not set")
            self._cohere_client = cohere.Client(api_key)
        return self._cohere_client
    
    def embed(
        self,
        texts: List[str],
        provider: str = "local",
        model: str = "all-MiniLM-L6-v2",
    ) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            provider: Embedding provider (local, openai, cohere)
            model: Model name
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings with {provider}/{model} for {len(texts)} texts")
        
        if provider == "local":
            return self._embed_local(texts, model)
        elif provider == "openai":
            return self._embed_openai(texts, model)
        elif provider == "cohere":
            return self._embed_cohere(texts, model)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def _embed_local(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using local sentence transformer"""
        model_obj = self._get_local_model(model)
        embeddings = model_obj.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def _embed_openai(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        client = self._get_openai_client()
        
        # Process in batches of 100
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_cohere(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using Cohere"""
        client = self._get_cohere_client()
        
        # Process in batches of 96 (Cohere limit)
        all_embeddings = []
        batch_size = 96
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embed(
                texts=batch,
                model=model,
                input_type="search_document",
            )
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
