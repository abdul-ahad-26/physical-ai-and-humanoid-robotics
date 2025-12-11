# API Contract: Index Endpoint

## Endpoint
`POST /index`

## Purpose
Accepts textbook content in Markdown/HTML format and indexes it for retrieval by the AI agent. This endpoint handles content ingestion, semantic chunking, and vector storage in Qdrant.

## Request

### Headers
- `Content-Type: application/json`
- `Authorization: Bearer <INDEXING_API_KEY>`

### Body Parameters
```json
{
  "content": {
    "type": "string",
    "required": true,
    "minLength": 1,
    "maxLength": 100000,
    "description": "Textbook content in Markdown or HTML format"
  },
  "metadata": {
    "type": "object",
    "required": true,
    "properties": {
      "source_file": {
        "type": "string",
        "required": true,
        "description": "Original filename of the content"
      },
      "section": {
        "type": "string",
        "required": false,
        "description": "Section identifier (e.g., chapter, heading)"
      },
      "document_type": {
        "type": "string",
        "enum": ["markdown", "html"],
        "required": false,
        "default": "markdown",
        "description": "Type of document format"
      }
    }
  }
}
```

### Example Request
```json
{
  "content": "# Chapter 7: Neural Networks\n\nNeural networks are computing systems inspired by the human brain...",
  "metadata": {
    "source_file": "chapter_7_neural_networks.md",
    "section": "Chapter 7",
    "document_type": "markdown"
  }
}
```

## Response

### Success Response (200 OK)
```json
{
  "status": {
    "type": "string",
    "enum": ["success", "partial", "queued"],
    "description": "Status of the indexing operation"
  },
  "indexed_chunks": {
    "type": "integer",
    "description": "Number of content chunks successfully indexed"
  },
  "content_id": {
    "type": "string",
    "description": "Unique identifier for the indexed content"
  },
  "processing_time": {
    "type": "number",
    "description": "Time taken to process the content in seconds"
  }
}
```

### Example Success Response
```json
{
  "status": "success",
  "indexed_chunks": 12,
  "content_id": "content_abc123",
  "processing_time": 2.45
}
```

### Error Responses

#### 401 Unauthorized
- **Condition**: Missing or invalid API key
- **Response Body**:
```json
{
  "error": "Unauthorized",
  "details": "Valid API key required for indexing"
}
```

#### 400 Bad Request
- **Condition**: Invalid request parameters
- **Response Body**:
```json
{
  "error": "Invalid input parameters",
  "details": "Error description"
}
```

#### 422 Unprocessable Entity
- **Condition**: Validation errors
- **Response Body**:
```json
{
  "error": "Validation failed",
  "details": {
    "content": "Content is required and cannot be empty"
  }
}
```

#### 429 Too Many Requests
- **Condition**: Rate limit exceeded for indexing
- **Response Body**:
```json
{
  "error": "Rate limit exceeded",
  "details": "Too many indexing requests, please try again later"
}
```

#### 503 Service Unavailable
- **Condition**: Qdrant vector database unavailable
- **Response Body**:
```json
{
  "error": "Service temporarily unavailable",
  "details": "Vector database is currently down, indexing failed"
}
```

## Security
- API key authentication required in Authorization header
- Rate limiting applied specifically for content indexing
- Input sanitization to prevent injection attacks
- Content validation to ensure proper format

## Performance Requirements
- Indexing operation should complete within 30 seconds for typical textbook sections
- Should handle content up to 100KB in size
- Semantic chunking should respect document structure

## Monitoring
- Log all indexing operations for tracking
- Monitor indexing success/failure rates
- Track processing times
- Record content types and volumes