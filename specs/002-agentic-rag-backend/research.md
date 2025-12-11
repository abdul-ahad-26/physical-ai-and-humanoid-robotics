# Research Summary: RAG + Agentic Backend for AI-Textbook Chatbot

## Decision: OpenAI Agents SDK with Gemini API Integration
**Rationale**: Using OpenAI Agents SDK as orchestration layer with Gemini API as model provider provides the best balance of advanced agentic capabilities and cost-effective model usage. The OpenAI SDK enables sophisticated tool usage, tracing, and orchestration while Gemini API provides high-quality responses with generous rate limits.

**Alternatives considered**:
- Native OpenAI GPT models with OpenAI SDK (higher cost)
- Anthropic Claude with custom orchestration (less agent-specific features)
- Open-source models with custom implementation (more complex maintenance)

## Decision: Qdrant Vector Database for RAG
**Rationale**: Qdrant provides excellent performance for similarity search, supports semantic chunking, and offers both cloud and local deployment options. It has strong Python client support and handles metadata efficiently.

**Alternatives considered**:
- Pinecone (managed but more expensive)
- Weaviate (good alternative but slightly less performant for our use case)
- FAISS (local only, less scalable)

## Decision: Hybrid Rate Limiting Approach
**Rationale**: Implementing request-based, token-based, and concurrent request limiting provides comprehensive protection against abuse while maintaining fair usage for legitimate users. This multi-layered approach handles different types of potential abuse patterns.

**Alternatives considered**:
- Simple request-based limiting (less sophisticated protection)
- Token-based only (doesn't account for request concurrency)
- Fixed window vs sliding window (chose sliding for better protection)

## Decision: Specialized Agent Architecture
**Rationale**: Having separate agents (MainOrchestrator, RAG, Indexing, Logging) provides clear separation of concerns, better maintainability, and allows for targeted optimization of each component. This follows clean architecture principles.

**Alternatives considered**:
- Single monolithic agent (harder to maintain and scale)
- Different specializations (settled on these four based on functional requirements)
- Microservice architecture (overkill for this project scope)

## Decision: Semantic/Context-Aware Chunking Strategy
**Rationale**: Respecting document structure (headings, paragraphs) while maintaining context coherence ensures that retrieved chunks are semantically meaningful and preserve the textbook's organizational structure, leading to better answers.

**Alternatives considered**:
- Fixed-size chunking (loses document structure)
- Sentence-based chunking (may break context coherence)
- Custom semantic boundaries (more complex to implement)

## Decision: Time-Based Data Retention Policy
**Rationale**: Implementing configurable retention (30/90/365 days) balances user privacy requirements with business needs for analytics and improvement. This approach is standard in the industry.

**Alternatives considered**:
- No retention policy (privacy concerns)
- Event-based retention (more complex to manage)
- User-controlled retention (requires additional UI/UX work)

## Decision: FastAPI HTTP Layer with Thin Routing
**Rationale**: FastAPI provides excellent performance, automatic OpenAPI documentation, async support, and type validation. Using it as a thin layer that forwards to agents maintains clean separation between HTTP concerns and business logic.

**Alternatives considered**:
- Flask (less performant, fewer built-in features)
- Django (overkill for API-only backend)
- Express.js (would require changing to Node.js ecosystem)