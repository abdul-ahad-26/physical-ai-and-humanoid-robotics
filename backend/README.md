# RAG + Agentic Backend for AI-Textbook Chatbot

This repository contains the backend service for a textbook-embedded AI assistant using a fully agentic architecture. The backend uses the OpenAI Agents SDK as the orchestration layer and the Gemini API as the model provider. The system includes content ingestion, RAG (Retrieval Augmented Generation) capabilities, specialized agents for different functions, and comprehensive logging with rate limiting.

## Features

- **Agentic Architecture**: Uses specialized agents (RAGAgent, IndexingAgent, LoggingAgent) coordinated by MainOrchestratorAgent
- **RAG System**: Content ingestion, semantic chunking, vector storage, and retrieval capabilities
- **Hybrid Rate Limiting**: Multiple strategies (request-based, token-based, concurrent requests)
- **Scalable Architecture**: Clean architecture principles with clear separation of concerns
- **Observability**: Structured logging, metrics, and health checks
- **Security**: API key protection, input sanitization, and rate limiting

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **AI/ML**: OpenAI Agents SDK, Google Generative AI (Gemini API)
- **Database**: Qdrant (vector database), Neon Postgres (session logging)
- **Testing**: pytest
- **Containerization**: Docker

## Prerequisites

- Python 3.11+
- Docker (optional, for containerization)
- Access to OpenAI and Google Generative AI APIs
- Qdrant vector database instance
- Neon Postgres database instance

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment variables file:
   ```bash
   cp .env.example .env
   ```

5. Update `.env` with your API keys and service URLs:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   NEON_DB_URL=your_neon_db_url_here
   INDEXING_API_KEY=your_indexing_api_key_here
   REDIS_URL=redis://localhost:6379  # Optional, for distributed rate limiting
   ```

## Usage

### Running locally

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running with Docker

```bash
# Build the Docker image
docker build -t rag-backend .

# Run the container
docker run -p 8000:8000 --env-file .env rag-backend
```

## API Endpoints

### Health Check
- `GET /api/v1/health` - Health check for all services
- `GET /api/v1/query/health` - Query service health
- `GET /api/v1/answer/health` - Answer service health
- `GET /api/v1/index/health` - Indexing service health

### Query
- `POST /api/v1/query` - Retrieve contexts for a query without generating an answer

**Request Body:**
```json
{
  "question": "What is machine learning?",
  "highlight_override": "Optional highlighted text to prioritize in search"
}
```

**Response:**
```json
{
  "retrieved_contexts": [
    {
      "content": "Content of the retrieved chunk...",
      "metadata": {
        "source_file": "ml_introduction.md",
        "section": "Introduction",
        "start_pos": 0,
        "end_pos": 500
      },
      "score": 0.85
    }
  ],
  "assembled_context": "Assembled context for LLM...",
  "query_id": "unique-query-id"
}
```

### Answer
- `POST /api/v1/answer` - Generate a natural language answer for a question

**Request Body:**
```json
{
  "question": "What is machine learning?",
  "k": 5,
  "highlight_override": "Optional highlighted text to prioritize in search"
}
```

**Parameters:**
- `question`: The user's question (required, 1-10000 characters)
- `k`: Number of content chunks to retrieve (optional, default: 3, range: 1-10)
- `highlight_override`: Optional highlighted text (optional, 1-5000 characters)

**Response:**
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved_contexts": [...],
  "confidence_score": 0.85,
  "answer_id": "unique-answer-id"
}
```

### Index
- `POST /api/v1/index` - Index content for retrieval (requires API key)

**Request Headers:**
```
Authorization: Bearer {INDEXING_API_KEY}
```

**Request Body:**
```json
{
  "content": "Markdown or HTML content",
  "source_file": "chapter1.md",
  "metadata": {
    "author": "Textbook Author",
    "subject": "Computer Science",
    "section": "Introduction"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "indexed_chunks": 15,
  "index_id": "unique-index-id"
}
```

### API Usage Examples

#### Python Client
```python
import requests

# Answer endpoint
response = requests.post(
    "http://localhost:8000/api/v1/answer",
    json={
        "question": "What is machine learning?",
        "k": 5
    }
)
answer_data = response.json()
print(f"Answer: {answer_data['answer']}")
print(f"Confidence: {answer_data['confidence_score']}")

# Query endpoint
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "What is artificial intelligence?",
        "highlight_override": "AI concepts"
    }
)
data = response.json()

# Index endpoint (requires authentication)
response = requests.post(
    "http://localhost:8000/api/v1/index",
    headers={
        "Authorization": f"Bearer {your_indexing_api_key}"
    },
    json={
        "content": "# Chapter 1: Introduction to AI...",
        "source_file": "chapter1.md",
        "metadata": {
            "author": "Author Name",
            "subject": "Computer Science"
        }
    }
)
```

#### cURL Examples
```bash
# Answer endpoint
curl -X POST http://localhost:8000/api/v1/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "k": 5
  }'

# Query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?",
    "highlight_override": "AI concepts"
  }'

# Health check
curl http://localhost:8000/api/v1/health
```

## Architecture

The backend follows a clean architecture with the following layers:

- **API Layer**: FastAPI HTTP endpoints
- **Orchestration Layer**: OpenAI Agents SDK with specialized agents
- **RAG Layer**: Content ingestion, chunking, and retrieval
- **Database Layer**: Qdrant for embeddings, Neon Postgres for logging
- **Tools Layer**: Specialized functions for different tasks

### Specialized Agents

- **MainOrchestratorAgent**: Coordinates between specialized agents
- **RAGAgent**: Handles retrieval-augmented generation tasks
- **IndexingAgent**: Manages content ingestion and indexing
- **LoggingAgent**: Handles session logging with time-based retention policy

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for orchestration | Yes |
| `GEMINI_API_KEY` | Google Generative AI API key | Yes |
| `QDRANT_URL` | URL for Qdrant vector database | Yes |
| `QDRANT_API_KEY` | API key for Qdrant (if required) | No |
| `NEON_DB_URL` | Connection string for Neon Postgres | Yes |
| `INDEXING_API_KEY` | API key for indexing endpoint | Yes |
| `REDIS_URL` | Redis URL for distributed rate limiting | No (optional) |

## Development

### Running Tests

```bash
# Run unit tests
python -m pytest tests/unit/

# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=src --cov-report=html
```

### Code Formatting

The project uses Black for code formatting:

```bash
# Check formatting
black --check src/

# Apply formatting
black src/
```

### Linting

The project uses flake8 for linting:

```bash
flake8 src/
```

## Deployment

The application can be deployed to platforms like Vercel, Render, or Railway. The Dockerfile is provided for containerized deployments.

### Environment Configuration

For production deployments, ensure you have appropriate environment variables set, particularly for:

- Database connection strings
- API keys (keep these secure)
- Rate limiting configuration
- Logging and monitoring endpoints

## Security

- All sensitive data should be stored in environment variables
- The indexing endpoint requires API key authentication
- Rate limiting is implemented to prevent abuse
- Input sanitization is performed on all user inputs
- Health check endpoints are publicly accessible but don't expose sensitive information

## Performance

The system is designed with performance in mind:

- Target retrieval latency: <300ms for top-5 search
- Target end-to-end answer time: <2s
- Agent tool call efficiency: 80% target
- Caching strategies implemented for frequent queries

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.