# Data Model: RAG + Agentic Backend for AI-Textbook Chatbot

## Entity: UserSession
**Purpose**: Represents a user interaction session with the AI chatbot

**Fields**:
- `id` (UUID): Unique identifier for the session
- `user_id` (String): Identifier for the user (optional for anonymous sessions)
- `query` (Text): The original question/query from the user
- `response` (Text): The AI-generated response
- `retrieved_context` (JSON): List of retrieved content chunks used to generate the response
- `timestamp` (DateTime): When the interaction occurred
- `session_metadata` (JSON): Additional metadata about the session (highlighted text, etc.)

**Relationships**:
- None (standalone entity for logging/analytics)

**Validation Rules**:
- `timestamp` must be in the past
- `query` and `response` must not exceed maximum length (e.g., 10,000 characters)
- `user_id` is optional for anonymous sessions

## Entity: TextbookContent
**Purpose**: Represents indexed textbook material with embeddings

**Fields**:
- `id` (UUID): Unique identifier for the content chunk
- `content` (Text): The actual content chunk
- `embeddings` (Vector): Embedding vector for semantic search
- `metadata` (JSON): Source information, document structure, etc.
  - `source_file`: Original file name
  - `section`: Document section (e.g., chapter, heading)
  - `page_number`: Page number if applicable
  - `document_type`: Markdown/HTML indicator
- `created_at` (DateTime): When the content was indexed
- `updated_at` (DateTime): When the content was last updated
- `status` (String): Active, deleted, or archived

**Relationships**:
- One-to-many with QueryContext (as source of retrieved content)

**Validation Rules**:
- `content` must not be empty
- `embeddings` must be properly formatted vector
- `status` must be one of: active, deleted, archived

## Entity: QueryContext
**Purpose**: Represents the context used for a specific query, including original question and retrieved content

**Fields**:
- `id` (UUID): Unique identifier for the query context
- `original_question` (Text): The user's original question
- `highlight_override` (Text): Optional highlighted text that replaces search context
- `retrieved_chunks` (JSON): List of content chunks retrieved from the knowledge base
- `processed_context` (Text): The final context sent to the AI model
- `timestamp` (DateTime): When the query was made
- `session_id` (UUID): Reference to the associated UserSession

**Relationships**:
- Belongs to UserSession
- References multiple TextbookContent chunks

**Validation Rules**:
- `original_question` must not be empty
- `retrieved_chunks` must contain valid content references

## Entity: AgentTool
**Purpose**: Represents a function/tool available to the AI agent

**Fields**:
- `id` (UUID): Unique identifier for the tool
- `name` (String): Name of the tool (e.g., "retrieve_context", "index_content", "log_session")
- `description` (Text): Description of what the tool does
- `parameters` (JSON): Schema defining the parameters the tool accepts
- `implementation_path` (String): File path to the tool implementation
- `active` (Boolean): Whether the tool is currently enabled

**Relationships**:
- Used by MainOrchestratorAgent, RAGAgent, IndexingAgent, LoggingAgent

**Validation Rules**:
- `name` must be unique
- `parameters` must be valid JSON Schema
- `implementation_path` must exist

## Entity: AgentExecutionLog
**Purpose**: Log of agent executions for debugging and monitoring

**Fields**:
- `id` (UUID): Unique identifier for the log entry
- `agent_id` (String): Identifier for which agent executed
- `tool_calls` (JSON): List of tools called during execution
- `input_params` (JSON): Parameters passed to the agent
- `output_result` (Text): Result returned by the agent
- `execution_time` (Float): Time taken in seconds
- `timestamp` (DateTime): When the execution occurred
- `session_id` (UUID): Reference to the associated UserSession

**Relationships**:
- Belongs to UserSession

**Validation Rules**:
- `agent_id` must be one of the defined agents
- `execution_time` must be positive

## Indexes and Performance Considerations

### TextbookContent Indexes
- Vector index on `embeddings` field for similarity search (Qdrant)
- Composite index on `metadata.source_file` and `created_at` for efficient retrieval by source
- Text search index on `content` for keyword search fallback

### UserSession Indexes
- Index on `timestamp` for time-based queries and retention cleanup
- Index on `user_id` for user-specific analytics (if user_id is provided)
- Composite index on `timestamp` and `user_id` for combined queries

### QueryContext Indexes
- Index on `session_id` for session-based retrieval
- Index on `timestamp` for time-based analysis

## Data Retention Policy

### UserSession Retention
- Automatic cleanup based on configurable policy (30/90/365 days)
- Cleanup job runs daily to remove expired sessions
- Sessions with status "deleted" are removed after 7 days in that state

### AgentExecutionLog Retention
- Keep for 30 days (shorter retention due to higher volume)
- Archive logs older than 7 days to long-term storage if needed

## Migration Strategy

### Initial Schema Creation
1. Create TextbookContent table with basic fields
2. Create UserSession table for logging
3. Create QueryContext table for context management
4. Create AgentTool table for tool registry
5. Create AgentExecutionLog table for monitoring

### Future Schema Changes
- Use versioned migration scripts
- Maintain backward compatibility during updates
- Implement shadow writing for new fields before switching reads