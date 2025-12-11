# ADR-004: Dual Database Storage Strategy

## Status
Accepted

## Date
2025-12-10

## Context
The system needs to store two fundamentally different types of data: high-dimensional embeddings for semantic search (RAG functionality) and structured session data for logging and analytics. We need to choose an appropriate storage strategy that optimizes for both use cases.

## Decision
We will use a dual database approach:
- **Qdrant Vector Database**: For storing embeddings and metadata for RAG functionality
  - Optimized for similarity search and vector operations
  - Supports semantic chunking and efficient retrieval
  - Offers both cloud and local deployment options
  - Strong Python client support

- **Neon Postgres Database**: For storing user session data and analytics
  - Optimized for structured data storage and complex queries
  - Supports time-based retention policies
  - Provides ACID transactions for data integrity
  - Connection pooling and transaction management

This approach allows each database to be optimized for its specific use case while maintaining clear separation of concerns.

## Consequences
**Positive**:
- Each database optimized for its specific use case
- Qdrant provides excellent performance for similarity search
- Postgres provides strong consistency for session data
- Clear separation of unstructured (vector) and structured data
- Can scale each database independently based on needs
- Supports time-based retention policies for session data

**Negative**:
- More complex deployment and operations
- Need to manage multiple database connections
- Potential for data consistency issues between systems
- More complex backup and recovery procedures
- Higher operational overhead

## Alternatives Considered
- **Single PostgreSQL database**: Would compromise RAG performance due to lack of vector search capabilities
- **Single vector database**: Would make structured session logging more complex and less efficient
- **NoSQL document store**: Would not provide the specialized capabilities needed for either use case
- **Single managed service**: Would limit flexibility and potentially compromise performance for one use case

## References
- specs/002-agentic-rag-backend/plan.md
- specs/002-agentic-rag-backend/research.md
- specs/002-agentic-rag-backend/data-model.md