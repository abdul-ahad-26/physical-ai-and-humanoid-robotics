# ADR-001: Backend Technology Stack and Architecture Pattern

## Status
Accepted

## Date
2025-12-10

## Context
We need to build a production-ready backend for an AI textbook chatbot that uses a fully agentic architecture. The system must support content ingestion, RAG (Retrieval Augmented Generation), specialized agents, and comprehensive logging. We need to select the technology stack and architectural approach that will support these requirements efficiently.

## Decision
We will use the following technology stack and architectural pattern:

**Backend Framework**: Python 3.11+ with FastAPI for the HTTP layer
- FastAPI provides excellent performance, automatic OpenAPI documentation, async support, and type validation
- Python is well-suited for AI/ML integration and has strong ecosystem support

**Agent Orchestration**: OpenAI Agents SDK with Gemini API as model provider
- OpenAI Agents SDK provides sophisticated orchestration capabilities and tool usage
- Gemini API provides high-quality responses with generous rate limits at lower cost than OpenAI alternatives

**Dependencies**: FastAPI, uvicorn, pydantic, openai, google-generativeai, qdrant-client, psycopg2-binary, python-dotenv

**Architecture Pattern**: Clean architecture with clear separation of concerns between agent, tools, RAG, database, and API layers
- Thin FastAPI HTTP layer that forwards to agents
- Specialized agents (MainOrchestrator, RAG, Indexing, Logging)
- Separate modules for each concern (agent, tools, rag, db, api)

## Consequences
**Positive**:
- Leverages Python's strong AI/ML ecosystem
- FastAPI provides excellent performance and developer experience
- OpenAI Agents SDK enables sophisticated agentic behavior
- Clean architecture enables maintainability and testability
- Gemini API provides cost-effective model usage
- Automatic OpenAPI documentation generation

**Negative**:
- Dependency on multiple external APIs (OpenAI, Google AI)
- Learning curve for OpenAI Agents SDK
- Potential vendor lock-in to specific AI provider patterns
- Need to manage multiple service integrations

## Alternatives Considered
- **Node.js with Express**: Would require changing to JavaScript ecosystem, less ideal for AI/ML
- **Django**: Overkill for API-only backend, would add unnecessary complexity
- **Flask**: Less performant, fewer built-in features than FastAPI
- **Native OpenAI models**: Higher cost than Gemini API
- **Anthropic Claude**: Less agent-specific features than OpenAI Agents SDK

## References
- specs/002-agentic-rag-backend/plan.md
- specs/002-agentic-rag-backend/research.md