# Knowledge Extraction Backend

FastAPI backend for document processing using IBM Docling with multi-provider embedding support.

## Features

- **IBM Docling Integration**: Process PDF, DOCX, PPTX, XLSX, HTML, audio, and images
- **Multi-Provider Embeddings**: OpenAI, Cohere, or local sentence transformers
- **Smart Chunking**: Preserve document structure for better retrieval
- **Upstash Vector**: Store and query embeddings

## Prerequisites

- Python 3.9-3.13
- pip or poetry

## Installation

### Local Development

\`\`\`bash
cd python-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Environment Variables

Create a `.env` file:

\`\`\`env
# Embedding providers (at least one required for cloud embeddings)
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# Upstash Vector
UPSTASH_VECTOR_REST_URL=https://...
UPSTASH_VECTOR_REST_TOKEN=...

# Optional: Default embedding provider
DEFAULT_EMBEDDING_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
\`\`\`

### Running Locally

\`\`\`bash
uvicorn main:app --reload --port 8000
\`\`\`

## Deployment

### Railway (Recommended)

1. Fork this repository
2. Create a new project in Railway
3. Connect to your forked repo
4. Set root directory to `python-backend`
5. Add environment variables
6. Deploy!

Railway will automatically detect the `Dockerfile` and deploy.

### Render

1. Create a new Web Service
2. Connect your repository
3. Set root directory to `python-backend`
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables
7. Deploy!

### Docker

\`\`\`bash
docker build -t knowledge-backend .
docker run -p 8000:8000 --env-file .env knowledge-backend
\`\`\`

## API Endpoints

### Health Check
\`\`\`
GET /health
\`\`\`

### Process Document
\`\`\`
POST /process
Content-Type: application/json

{
  "document_id": "abc123",
  "blob_url": "https://...",
  "embedding_provider": "openai",  // optional
  "embedding_model": "text-embedding-3-small"  // optional
}
\`\`\`

### Get Processing Status
\`\`\`
GET /status/{document_id}
\`\`\`

## Supported File Types

| Category   | Formats              |
|------------|----------------------|
| Documents  | PDF, DOCX, PPTX, XLSX|
| Web        | HTML                 |
| Audio      | WAV, MP3             |
| Images     | PNG, JPEG, TIFF      |
