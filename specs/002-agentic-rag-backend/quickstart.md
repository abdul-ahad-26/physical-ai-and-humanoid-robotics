# Quickstart Guide: RAG + Agentic Backend for AI-Textbook Chatbot

## Prerequisites

- Python 3.11+
- pip package manager
- Git
- Docker (for local Qdrant and Neon Postgres if running locally)
- API keys for:
  - OpenAI (for agent orchestration and tracing)
  - Google AI/Gemini (for model responses)
  - Qdrant Cloud (or local instance)
  - Neon Postgres (or local instance)

## Local Development Setup

### 1. Clone and Navigate to Project
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Copy the example environment file and configure your API keys:
```bash
cp .env.example .env
# Edit .env with your actual API keys and service URLs
```

Required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DB_URL=your_neon_db_url
INDEXING_API_KEY=your_secret_api_key_for_indexing_endpoint
```

### 5. Start Required Services

For local development, you can run Qdrant and Postgres using Docker:

```bash
# Start Qdrant
docker run -d --name qdrant-container -p 6333:6333 qdrant/qdrant

# Start Postgres (for local development only)
docker run -d --name postgres-local -p 5432:5432 -e POSTGRES_DB=textbook_chat -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password postgres:13
```

## Running the Application

### 1. Start the Backend Server
```bash
# Activate virtual environment
source venv/bin/activate

# Start the FastAPI server
uvicorn backend.src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Verify Health
Check that all services are running:
```
GET http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "fastapi": "ok",
    "qdrant": "ok",
    "neon": "ok"
  }
}
```

## Basic Usage Examples

### 1. Index Textbook Content
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_INDEXING_API_KEY" \
  -d '{
    "content": "# Introduction to AI\n\nArtificial Intelligence is a branch of computer science...",
    "metadata": {
      "source_file": "intro_ai.md",
      "section": "Chapter 1"
    }
  }'
```

### 2. Query for Information
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "highlight_override": null
  }'
```

### 3. Get an Answer from the Agent
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the key concepts of machine learning",
    "k": 3,
    "highlight_override": null
  }'
```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=backend
```

### Code Formatting
```bash
# Format code with black
black backend/

# Lint code with flake8
flake8 backend/

# Type checking with mypy
mypy backend/
```

### Adding New Content
1. Prepare your textbook content in Markdown or HTML format
2. Use the `/index` endpoint with proper authentication to add content
3. Verify the content was indexed by querying for known terms

### Debugging Tips
- Check the agent execution logs for tool call information
- Use the `/health` endpoint to verify service connectivity
- Monitor response times to identify performance bottlenecks
- Enable detailed logging by setting `LOG_LEVEL=DEBUG` in your environment

## Production Deployment

### Environment Variables for Production
In addition to the development variables, production deployments should set:
```
LOG_LEVEL=INFO
DEBUG=False
DATABASE_POOL_SIZE=20
MAX_WORKERS=10
```

### Deployment to Vercel/Render/Railway
1. Set up your environment variables in the deployment platform
2. Ensure your Qdrant and Neon services are accessible from the deployment environment
3. Deploy using the platform's standard Python deployment process

### Scaling Considerations
- Monitor API usage and scale accordingly
- Consider implementing caching for frequently requested content
- Set up proper monitoring and alerting for service health