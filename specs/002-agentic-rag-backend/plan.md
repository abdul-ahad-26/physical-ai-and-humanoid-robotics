# Implementation Plan: RAG + Agentic Backend for AI-Textbook Chatbot

**Branch**: `002-agentic-rag-backend` | **Date**: 2025-12-10 | **Spec**: [specs/002-agentic-rag-backend/spec.md](specs/002-agentic-rag-backend/spec.md)
**Input**: Feature specification from `/specs/002-agentic-rag-backend/spec.md`

## Summary

This plan outlines the implementation of a production-ready backend that powers a textbook-embedded AI assistant using a fully agentic architecture. The backend uses the OpenAI Agents SDK as the orchestration layer and the Gemini API as the model provider. The system includes content ingestion, RAG (Retrieval Augmented Generation) capabilities, specialized agents for different functions, and comprehensive logging with rate limiting.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, google-generativeai, qdrant-client, psycopg2-binary, python-dotenv
**Storage**: Qdrant vector database for embeddings, Neon Postgres for session logging
**Testing**: pytest for unit/integration testing
**Target Platform**: Linux server (deployable on Vercel, Render, or Railway)
**Project Type**: backend service (web)
**Performance Goals**: Retrieval latency < 300ms for top-5 search, End-to-end answer < 2s
**Constraints**: <200ms p95 for retrieval, Agent must minimize unnecessary tool calls (80% efficiency)
**Scale/Scope**: Support multiple concurrent users with hybrid rate limiting

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Architecture follows clean architecture principles with clear separation of concerns between agent, tools, RAG, database, and API layers. All dependencies are properly licensed and security considerations are addressed with API key protection and input sanitization.

## Project Structure

### Documentation (this feature)

```text
specs/002-agentic-rag-backend/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # MainOrchestratorAgent implementation
│   │   ├── rag_agent.py            # RAGAgent implementation
│   │   ├── indexing_agent.py       # IndexingAgent implementation
│   │   └── logging_agent.py        # LoggingAgent implementation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tools.py            # RAG retrieval tools
│   │   ├── indexing_tools.py       # Content indexing tools
│   │   └── logging_tools.py        # Session logging tools
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py              # Semantic/content-aware chunking
│   │   ├── embedder.py             # Embedding generation using free model
│   │   └── retriever.py            # Qdrant-based retrieval
│   ├── db/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py        # Qdrant vector database client
│   │   ├── postgres_client.py      # Neon Postgres client
│   │   └── models.py               # Database models
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app entry point
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── query.py            # Query endpoint
│       │   ├── answer.py           # Answer endpoint
│       │   ├── index.py            # Index endpoint
│       │   └── health.py           # Health check endpoint
│       └── middleware/
│           ├── __init__.py
│           ├── rate_limiter.py     # Hybrid rate limiting middleware
│           └── auth.py             # API key authentication middleware
├── tests/
│   ├── unit/
│   │   ├── agent/
│   │   ├── tools/
│   │   ├── rag/
│   │   ├── db/
│   │   └── api/
│   ├── integration/
│   │   ├── agent/
│   │   ├── tools/
│   │   ├── rag/
│   │   ├── db/
│   │   └── api/
│   └── contract/
│       └── api_contracts.py
├── requirements.txt
├── .env.example
├── .env
└── README.md
```

**Structure Decision**: Backend service with clean architecture separating concerns into agent, tools, RAG, database, and API layers. This structure supports the specialized agent architecture with clear boundaries between components.

## Architecture Breakdown

### 1. FastAPI HTTP Layer
- **Purpose**: Thin routing layer that forwards requests to the agent
- **Components**:
  - Main API application with health, query, answer, and index endpoints
  - Middleware for authentication and rate limiting
  - Request/response validation models
- **Dependencies**: FastAPI, uvicorn, pydantic
- **Error Handling**: Graceful degradation when downstream services unavailable

### 2. OpenAI Agents SDK Orchestration Layer
- **Purpose**: Central orchestration using OpenAI Agents SDK
- **Components**:
  - MainOrchestratorAgent: Coordinates between specialized agents
  - RAGAgent: Handles retrieval-augmented generation tasks
  - IndexingAgent: Manages content ingestion and indexing
  - LoggingAgent: Handles session logging and analytics
- **Dependencies**: openai, custom model provider for Gemini
- **Error Handling**: Fallback strategies when primary agent fails

### 3. Gemini Model Provider Integration
- **Purpose**: Use Gemini API as the LLM for the agent
- **Components**:
  - Custom model provider adapter for Gemini
  - API key management for observability
  - Request/response transformation
- **Dependencies**: google-generativeai, python-dotenv
- **Security**: API keys stored in environment variables

### 4. Qdrant Vector Database for RAG
- **Purpose**: Store embeddings and metadata for retrieval
- **Components**:
  - Qdrant client for vector operations
  - Collection management for textbook content
  - Search and similarity functions
- **Dependencies**: qdrant-client, grpcio
- **Error Handling**: Graceful degradation when Qdrant unavailable

### 5. Neon Postgres for Logging
- **Purpose**: Store user session data and analytics
- **Components**:
  - Database models for UserSession, TextbookContent, QueryContext
  - Connection pooling and transaction management
  - Data retention cleanup jobs
- **Dependencies**: psycopg2-binary, SQLAlchemy
- **Security**: Input sanitization to prevent injection attacks

### 6. Specialized Agents
- **MainOrchestratorAgent**: Coordinates between specialized agents and manages workflow
- **RAGAgent**: Handles retrieval tasks and content retrieval
- **IndexingAgent**: Manages content ingestion, chunking, and vector operations
- **LoggingAgent**: Handles session logging with time-based retention policy

### 7. Skills/Tools
- **RAG Retrieval Chains**: Semantic searches, content retrieval, result ranking
- **Vector Indexing**: Content chunking, embedding generation, vector database operations
- **Data Validation**: HTML/Markdown input sanitization, content format validation
- **Backend Logic**: API endpoint handling, request/response processing
- **Security & Rate Limiting**: API key validation, hybrid rate limiting, abuse prevention

## Implementation Roadmap

### Phase 1: Foundation Setup
1. **Project Structure Setup**
   - Create directory structure as defined above
   - Set up requirements.txt with all dependencies
   - Configure environment variables and .env.example
   - Set up basic FastAPI application structure

2. **Database Layer Implementation**
   - Implement Qdrant client for vector database
   - Create database models for Neon Postgres
   - Implement connection pooling and basic operations
   - Add data retention cleanup jobs

3. **Basic RAG Pipeline**
   - Implement semantic/content-aware chunking
   - Create embedding generation using free model
   - Build basic retrieval functionality
   - Add error handling for Qdrant unavailability

### Phase 2: Agent Architecture
1. **Specialized Agent Implementation**
   - Build MainOrchestratorAgent with hybrid rate limiting
   - Implement RAGAgent for retrieval tasks
   - Create IndexingAgent for content ingestion
   - Develop LoggingAgent for session logging

2. **Tool Development**
   - Create RAG retrieval tools with semantic search
   - Build vector indexing tools for content management
   - Implement data validation tools for input sanitization
   - Develop security tools for rate limiting and authentication

### Phase 3: API Layer and Integration
1. **HTTP API Implementation**
   - Create query endpoint with highlight override support
   - Implement answer endpoint with hybrid rate limiting
   - Build index endpoint with API key protection
   - Add health check endpoints for all services

2. **Integration and Testing**
   - Connect agents to API endpoints
   - Implement comprehensive error handling
   - Add observability with logging and tracing
   - Create thorough test coverage

### Phase 4: Production Readiness
1. **Performance Optimization**
   - Optimize retrieval latency for <300ms
   - Improve agent tool call efficiency to 80%
   - Implement caching strategies where appropriate

2. **Security and Observability**
   - Finalize rate limiting strategies
   - Complete security hardening
   - Add comprehensive monitoring and alerting
   - Prepare deployment configurations

## Sequence of Development

1. **Week 1**: Project setup, database layer, basic RAG pipeline
2. **Week 2**: Specialized agent implementation, tool development
3. **Week 3**: API layer, integration, testing
4. **Week 4**: Performance optimization, security, deployment prep

## Agent-Level Plan

### MainOrchestratorAgent
- **Responsibilities**: Coordinate between specialized agents, manage workflow, implement rate limiting
- **Tools**: Access to all specialized agent tools
- **Invocation**: Triggered by API endpoints, manages agent handoffs
- **Failure Modes**: Fallback to specialized agents directly, graceful degradation

### RAGAgent
- **Responsibilities**: Handle retrieval-augmented generation tasks, content retrieval using semantic chunking
- **Tools**: RAG retrieval tools, embedding generation, Qdrant operations
- **Invocation**: Called by orchestrator when retrieval needed
- **Failure Modes**: Return minimal context, graceful degradation

### IndexingAgent
- **Responsibilities**: Content ingestion in Markdown/HTML, semantic chunking, vector database operations
- **Tools**: Content parsing, chunking, embedding generation, Qdrant operations
- **Invocation**: Called by orchestrator for content indexing
- **Failure Modes**: Rollback partial indexing, maintain data consistency

### LoggingAgent
- **Responsibilities**: Session logging with time-based retention, analytics collection
- **Tools**: Neon Postgres operations, data retention cleanup
- **Invocation**: Called by orchestrator after interactions
- **Failure Modes**: Continue operation without logging, maintain functionality

## RAG Pipeline Plan

### Content Ingestion
- Accept Markdown/HTML content from textbook
- Parse and validate content format
- Apply semantic/content-aware chunking rules
- Generate embeddings using free model
- Store in Qdrant with metadata

### Semantic Chunking Rules
- Respect document structure (headings, paragraphs)
- Maintain context coherence within chunks
- Limit chunk size to appropriate token limits
- Preserve important metadata and relationships

### Embedding Generation
- Use free embedding model (e.g., text-embedding-3-small)
- Generate embeddings for content chunks
- Store alongside metadata in Qdrant
- Support incremental updates

### Index Creation/Update in Qdrant
- Create collections for different textbook content
- Implement upsert operations for updates
- Support deletion and re-indexing
- Handle schema evolution for metadata

### Retrieval Flow
- Embed user query using same model as content
- Search Qdrant for top-K similar chunks
- Rank and return relevant content with metadata
- Support highlight override mode

### Fallback Mode When Qdrant Down
- Return limited functionality notice
- Maintain basic API availability
- Log failure for observability
- Implement circuit breaker pattern

## Data & Storage Plan

### Database Models
- **UserSession**: Store query, response, timestamp, user ID with retention policy
- **TextbookContent**: Store chunked content, embeddings, metadata, source information
- **QueryContext**: Store original question, highlighted text override, retrieved snippets

### Retention Logic
- Implement time-based cleanup (30/90/365 days)
- Schedule cleanup jobs using cron or similar
- Maintain referential integrity during cleanup
- Log cleanup operations for audit purposes

### Connection Management
- Implement connection pooling for Neon Postgres
- Use async operations where possible
- Handle connection failures gracefully
- Implement retry logic with exponential backoff

## API Design Plan

### POST /query
- **Input**: {question, highlight_override?}
- **Output**: retrieved contexts + assembled prompt context
- **Auth**: None (public endpoint)
- **Rate Limiting**: Hybrid approach (requests/tokens/concurrent)
- **Error Handling**: Return appropriate HTTP codes, fallback when Qdrant down

### POST /answer
- **Input**: {question, k, highlight_override?}
- **Output**: final natural-language answer from agent
- **Auth**: None (public endpoint)
- **Rate Limiting**: Full hybrid rate limiting
- **Error Handling**: Comprehensive error responses, fallback options

### POST /index
- **Input**: markdown/html payload
- **Auth**: API key required in header
- **Rate Limiting**: Content-based limits
- **Error Handling**: Validation errors, content processing failures

### GET /health
- **Output**: Health status of FastAPI, Qdrant, Neon
- **Auth**: None
- **Error Handling**: Component-specific health indicators

## Deployment & DevOps

### Environments
- **Development**: Local Docker setup with local Qdrant
- **Staging**: Cloud deployment with test databases
- **Production**: Vercel/Render/Railway with production databases

### Runtime Configuration
- Environment variables for all external service URLs and credentials
- Configuration files for rate limiting parameters
- Feature flags for gradual rollout of new capabilities

### Secret Management
- Environment variables for API keys
- Secure storage for database credentials
- Rotation strategy for API keys

### Observability
- Structured logging with correlation IDs
- Performance metrics for key operations
- Error tracking and alerting
- OpenTelemetry integration for tracing

## Mapping to Tasks

1. **Project Setup**: Create directory structure, dependencies, environment configuration
2. **Database Layer**: Implement Qdrant and Postgres clients with models
3. **RAG Pipeline**: Build content ingestion, chunking, and retrieval
4. **Agent Architecture**: Implement specialized agents and their coordination
5. **API Layer**: Create HTTP endpoints and middleware
6. **Integration**: Connect all components and implement error handling
7. **Testing**: Unit, integration, and contract tests
8. **Performance**: Optimize for latency and efficiency targets
9. **Security**: Rate limiting, input validation, API key protection
10. **Deployment**: Prepare configurations for cloud deployment

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple database systems (Qdrant + Postgres) | Need vector search for RAG and structured logging | Single system would compromise either RAG performance or logging structure |
| Multiple specialized agents | Required for clean separation of concerns | Single agent would create monolithic, hard-to-maintain code |
