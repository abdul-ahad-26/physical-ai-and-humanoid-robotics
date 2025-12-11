# API Contract: Query Endpoint

## Endpoint
`POST /query`

## Purpose
Accepts a user question and optional highlight override, returns retrieved contexts and assembled prompt context without generating a final answer. This endpoint is designed to provide the raw context that would be sent to the AI model.

## Request

### Headers
- `Content-Type: application/json`

### Body Parameters
```json
{
  "question": {
    "type": "string",
    "required": true,
    "minLength": 1,
    "maxLength": 10000,
    "description": "The user's question about textbook content"
  },
  "highlight_override": {
    "type": "string",
    "required": false,
    "minLength": 1,
    "maxLength": 5000,
    "description": "Optional highlighted text that replaces search context"
  }
}
```

### Example Request
```json
{
  "question": "What are the main principles of machine learning?",
  "highlight_override": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience."
}
```

## Response

### Success Response (200 OK)
```json
{
  "retrieved_contexts": {
    "type": "array",
    "items": {
      "content": "string",
      "metadata": "object",
      "score": "number"
    }
  },
  "assembled_context": {
    "type": "string",
    "description": "The combined context that would be sent to the AI model"
  },
  "query_id": {
    "type": "string",
    "description": "Unique identifier for this query operation"
  }
}
```

### Example Success Response
```json
{
  "retrieved_contexts": [
    {
      "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
      "metadata": {
        "source_file": "chapter_3_ml_fundamentals.md",
        "section": "3.1 Introduction to ML",
        "page_number": 45
      },
      "score": 0.87
    },
    {
      "content": "The three main types of machine learning are supervised, unsupervised, and reinforcement learning.",
      "metadata": {
        "source_file": "chapter_3_ml_fundamentals.md",
        "section": "3.2 Types of ML",
        "page_number": 47
      },
      "score": 0.78
    }
  ],
  "assembled_context": "Based on the textbook content, machine learning is a subset of AI that enables systems to learn from experience. The main types are supervised, unsupervised, and reinforcement learning...",
  "query_id": "query_12345"
}
```

### Error Responses

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
    "question": "Question is required and cannot be empty"
  }
}
```

#### 503 Service Unavailable
- **Condition**: Qdrant vector database unavailable
- **Response Body**:
```json
{
  "error": "Service temporarily unavailable",
  "details": "Vector database is currently down, using limited functionality"
}
```

## Security
- No authentication required (public endpoint)
- Rate limiting applied using hybrid approach (requests, tokens, concurrent)

## Performance Requirements
- Response time: < 300ms for 95th percentile
- Should gracefully degrade when Qdrant is unavailable

## Monitoring
- Log all requests for analytics
- Track retrieval success/failure rates
- Monitor response times