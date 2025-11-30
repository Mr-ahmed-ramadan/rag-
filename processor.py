"""
Document processor using IBM Docling
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Docling
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Install with: pip install docling")


class DocumentProcessor:
    """Process documents using IBM Docling"""
    
    def __init__(self):
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()
        else:
            self.converter = None
    
    def is_available(self) -> bool:
        """Check if Docling is available"""
        return DOCLING_AVAILABLE and self.converter is not None
    
    def process(self, content: bytes, file_ext: str, document_id: str) -> Dict[str, Any]:
        """
        Process document content and extract text with structure
        
        Args:
            content: Raw file bytes
            file_ext: File extension (pdf, docx, etc.)
            document_id: Unique document identifier
            
        Returns:
            Extracted content with metadata
        """
        if not self.is_available():
            raise RuntimeError("Docling is not available")
        
        # Write content to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Convert document
            result = self.converter.convert(tmp_path)
            
            # Extract text and structure
            doc_dict = result.document.export_to_dict()
            
            return {
                "document_id": document_id,
                "content": result.document.export_to_markdown(),
                "structure": doc_dict,
                "metadata": {
                    "title": doc_dict.get("name", ""),
                    "page_count": len(doc_dict.get("pages", [])),
                    "file_type": file_ext,
                }
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def chunk(
        self,
        extracted: Dict[str, Any],
        document_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Chunk extracted content with metadata preservation
        
        Args:
            extracted: Extracted document data
            document_id: Document identifier
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        content = extracted["content"]
        chunks = []
        
        # Simple chunking by paragraphs with size limits
        paragraphs = content.split("\n\n")
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size:
                if current_chunk:
                    chunks.append({
                        "id": f"{document_id}_{chunk_index}",
                        "content": current_chunk.strip(),
                        "metadata": {
                            "document_id": document_id,
                            "chunk_index": chunk_index,
                            "start_offset": current_start,
                            "end_offset": current_start + len(current_chunk),
                            "file_type": extracted["metadata"]["file_type"],
                            "title": extracted["metadata"]["title"],
                        }
                    })
                    chunk_index += 1
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                    current_start = current_start + overlap_start
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "id": f"{document_id}_{chunk_index}",
                "content": current_chunk.strip(),
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "start_offset": current_start,
                    "end_offset": current_start + len(current_chunk),
                    "file_type": extracted["metadata"]["file_type"],
                    "title": extracted["metadata"]["title"],
                    "total_chunks": chunk_index + 1,
                }
            })
        
        # Update total_chunks in all metadata
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
