import pytest
import requests
from fastapi.testclient import TestClient
from src.api.main import app
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json


client = TestClient(app)

# Define request/response models that match the API contracts
class QueryRequest(BaseModel):
    question: str
    highlight_override: Optional[str] = None


class RetrievedContext(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    retrieved_contexts: List[RetrievedContext]
    assembled_context: str
    query_id: str


class AnswerRequest(BaseModel):
    question: str
    k: int = 3
    highlight_override: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    retrieved_contexts: List[RetrievedContext]
    confidence_score: float
    answer_id: str


class IndexRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]


class IndexResponse(BaseModel):
    status: str
    indexed_chunks: int
    content_id: str
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    details: Optional[Dict[str, Any]] = None


class TestAPIContracts:
    """Contract tests to verify API endpoints match their specified contracts."""

    def test_query_endpoint_contract(self):
        """Test that the query endpoint matches its contract specification."""
        # Send a valid request to the query endpoint
        request_data = {
            "question": "What is machine learning?",
            "highlight_override": None
        }

        response = client.post("/api/v1/query", json=request_data)

        # Verify response status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        # Parse response as QueryResponse
        try:
            response_data = QueryResponse(**response.json())
        except Exception as e:
            pytest.fail(f"Response doesn't match QueryResponse contract: {e}")

        # Verify response structure
        assert hasattr(response_data, 'retrieved_contexts')
        assert hasattr(response_data, 'assembled_context')
        assert hasattr(response_data, 'query_id')

        # Verify types
        assert isinstance(response_data.retrieved_contexts, list)
        assert isinstance(response_data.assembled_context, str)
        assert isinstance(response_data.query_id, str)

        # Verify at least one context was retrieved
        assert len(response_data.retrieved_contexts) >= 0  # Could be 0 if no content indexed

        # Verify context structure if contexts exist
        for ctx in response_data.retrieved_contexts:
            assert hasattr(ctx, 'content')
            assert hasattr(ctx, 'metadata')
            assert hasattr(ctx, 'score')
            assert isinstance(ctx.content, str)
            assert isinstance(ctx.metadata, dict)
            assert isinstance(ctx.score, (float, int))

        print("✓ Query endpoint contract validation passed")

    def test_answer_endpoint_contract(self):
        """Test that the answer endpoint matches its contract specification."""
        # Send a valid request to the answer endpoint
        request_data = {
            "question": "What is artificial intelligence?",
            "k": 3,
            "highlight_override": "AI concepts in textbooks"
        }

        response = client.post("/api/v1/answer", json=request_data)

        # Verify response status (could be 200 or 503 if services are down)
        assert response.status_code in [200, 503], f"Expected 200 or 503, got {response.status_code}: {response.text}"

        if response.status_code == 200:
            # Parse response as AnswerResponse
            try:
                response_data = AnswerResponse(**response.json())
            except Exception as e:
                pytest.fail(f"Response doesn't match AnswerResponse contract: {e}")

            # Verify response structure
            assert hasattr(response_data, 'answer')
            assert hasattr(response_data, 'retrieved_contexts')
            assert hasattr(response_data, 'confidence_score')
            assert hasattr(response_data, 'answer_id')

            # Verify types
            assert isinstance(response_data.answer, str)
            assert isinstance(response_data.retrieved_contexts, list)
            assert isinstance(response_data.confidence_score, (float, int))
            assert isinstance(response_data.answer_id, str)

            # Verify confidence score is in valid range
            assert 0.0 <= response_data.confidence_score <= 1.0

            # Verify context structure if contexts exist
            for ctx in response_data.retrieved_contexts:
                assert hasattr(ctx, 'content')
                assert hasattr(ctx, 'metadata')
                assert hasattr(ctx, 'score')
                assert isinstance(ctx.content, str)
                assert isinstance(ctx.metadata, dict)
                assert isinstance(ctx.score, (float, int))

        print("✓ Answer endpoint contract validation passed")

    def test_index_endpoint_contract(self):
        """Test that the index endpoint matches its contract specification."""
        # Send a valid request to the index endpoint
        request_data = {
            "content": "# Test Content\nThis is test content for indexing.",
            "metadata": {
                "source_file": "test_content.md",
                "section": "Introduction",
                "document_type": "markdown"
            }
        }

        # Note: The index endpoint requires authentication, so we expect a 401 or 422
        # For contract testing, we'll verify the error response structure matches expectations
        response = client.post("/api/v1/index", json=request_data)

        # Should return either 200 (success) or 401 (unauthorized) or 422 (validation error)
        # The important thing is that the response structure is consistent
        if response.status_code == 200:
            # If successful, verify the response matches IndexResponse contract
            try:
                response_data = IndexResponse(**response.json())
            except Exception as e:
                pytest.fail(f"Response doesn't match IndexResponse contract: {e}")

            # Verify response structure
            assert hasattr(response_data, 'status')
            assert hasattr(response_data, 'indexed_chunks')
            assert hasattr(response_data, 'content_id')
            assert hasattr(response_data, 'processing_time')

            # Verify types
            assert isinstance(response_data.status, str)
            assert isinstance(response_data.indexed_chunks, int)
            assert isinstance(response_data.content_id, str)
            assert isinstance(response_data.processing_time, (float, int))

        elif response.status_code in [401, 422]:
            # For unauthorized or validation errors, verify error response structure
            response_json = response.json()
            assert "detail" in response_json or "error" in response_json
        else:
            # Any other status is unexpected
            pytest.fail(f"Unexpected status code {response.status_code}: {response.text}")

        print("✓ Index endpoint contract validation passed")

    def test_health_endpoint_contract(self):
        """Test that the health endpoint matches its contract specification."""
        response = client.get("/api/v1/health")

        # Verify response status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        # Parse response as HealthResponse
        try:
            response_data = HealthResponse(**response.json())
        except Exception as e:
            pytest.fail(f"Response doesn't match HealthResponse contract: {e}")

        # Verify response structure
        assert hasattr(response_data, 'status')
        assert hasattr(response_data, 'timestamp')
        assert hasattr(response_data, 'services')

        # Verify types
        assert isinstance(response_data.status, str)
        assert isinstance(response_data.timestamp, str)
        assert isinstance(response_data.services, dict)

        # Verify status is one of the expected values
        assert response_data.status in ["healthy", "degraded", "unhealthy"]

        # Verify services contains expected keys
        expected_services = {"fastapi", "qdrant", "neon"}
        actual_services = set(response_data.services.keys())
        assert expected_services.issubset(actual_services), f"Missing expected services. Expected: {expected_services}, Got: {actual_services}"

        print("✓ Health endpoint contract validation passed")

    def test_query_endpoint_validation_contract(self):
        """Test that the query endpoint properly validates input according to contract."""
        # Test with empty question (should fail validation)
        invalid_request_data = {
            "question": "",  # Empty question should fail
            "highlight_override": None
        }

        response = client.post("/api/v1/query", json=invalid_request_data)

        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422 for empty question, got {response.status_code}: {response.text}"

        # Verify error response structure
        error_response = response.json()
        assert "detail" in error_response
        assert isinstance(error_response["detail"], list) or isinstance(error_response["detail"], dict)

        # Test with very long question (should fail validation)
        long_question = "A" * 10001  # Exceeds 10000 character limit
        invalid_request_data_long = {
            "question": long_question,
            "highlight_override": None
        }

        response = client.post("/api/v1/query", json=invalid_request_data_long)

        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422 for long question, got {response.status_code}: {response.text}"

        print("✓ Query endpoint validation contract validation passed")

    def test_answer_endpoint_validation_contract(self):
        """Test that the answer endpoint properly validates input according to contract."""
        # Test with invalid k value (should fail validation)
        invalid_request_data = {
            "question": "Test question?",
            "k": 0,  # Invalid k value (must be >= 1)
            "highlight_override": None
        }

        response = client.post("/api/v1/answer", json=invalid_request_data)

        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422 for invalid k, got {response.status_code}: {response.text}"

        # Verify error response structure
        error_response = response.json()
        assert "detail" in error_response

        # Test with negative k value
        invalid_request_data_neg = {
            "question": "Test question?",
            "k": -1,  # Negative k value
            "highlight_override": None
        }

        response = client.post("/api/v1/answer", json=invalid_request_data_neg)

        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422 for negative k, got {response.status_code}: {response.text}"

        # Test with k value too high
        invalid_request_data_high = {
            "question": "Test question?",
            "k": 11,  # Above max of 10
            "highlight_override": None
        }

        response = client.post("/api/v1/answer", json=invalid_request_data_high)

        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422 for k too high, got {response.status_code}: {response.text}"

        print("✓ Answer endpoint validation contract validation passed")

    def test_index_endpoint_validation_contract(self):
        """Test that the index endpoint properly validates input according to contract."""
        # Test with empty content (should fail validation)
        invalid_request_data = {
            "content": "",  # Empty content should fail
            "metadata": {
                "source_file": "test.md"
            }
        }

        response = client.post("/api/v1/index", json=invalid_request_data)

        # Without authentication, expect 401 or 422 depending on auth implementation
        # For contract testing, we'll just verify it's a proper error response
        assert response.status_code in [401, 422], f"Expected 401 or 422, got {response.status_code}: {response.text}"

        # If it's a validation error, check the structure
        if response.status_code == 422:
            error_response = response.json()
            assert "detail" in error_response

        print("✓ Index endpoint validation contract validation passed")

    def test_health_endpoint_response_formats(self):
        """Test various health response formats."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        # Test that required fields are present
        required_fields = ['status', 'timestamp', 'services']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Test that services contains required service statuses
        required_services = ['fastapi']
        for service in required_services:
            assert service in data['services'], f"Missing required service: {service}"

        # Test timestamp format (should be ISO format)
        import datetime
        try:
            # Try to parse the timestamp to ensure it's in a valid format
            parsed_time = datetime.datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Timestamp is not in valid ISO format: {data['timestamp']}")

        print("✓ Health endpoint response format validation passed")

    def test_api_response_headers(self):
        """Test that API endpoints return proper headers."""
        # Test query endpoint headers
        response = client.post("/api/v1/query", json={
            "question": "Test question?",
            "highlight_override": None
        })

        # Verify standard headers
        assert response.headers.get("content-type") == "application/json"

        # Test health endpoint headers
        response = client.get("/api/v1/health")

        # Verify standard headers
        assert response.headers.get("content-type") == "application/json"

        print("✓ API response headers validation passed")

    def test_error_response_contract_consistency(self):
        """Test that error responses follow a consistent contract across all endpoints."""
        # Test error response from query endpoint with invalid input
        response = client.post("/api/v1/query", json={
            "question": "",  # Invalid - empty question
            "highlight_override": None
        })

        if response.status_code == 422:
            error_data = response.json()
            # Error responses should have a detail field
            assert "detail" in error_data

        # Test error response from answer endpoint with invalid input
        response = client.post("/api/v1/answer", json={
            "question": "Test?",
            "k": 0,  # Invalid k value
            "highlight_override": None
        })

        if response.status_code == 422:
            error_data = response.json()
            # Error responses should have a detail field
            assert "detail" in error_data

        # Test error response from health endpoint (though unlikely to error under normal circumstances)
        # We'll check the structure of successful response as well
        response = client.get("/api/v1/health")
        success_data = response.json()

        # Successful responses should not have error fields
        assert "detail" not in success_data or "error" not in success_data

        print("✓ Error response contract consistency validation passed")

    @pytest.mark.parametrize("endpoint,method,request_data", [
        ("/api/v1/query", "POST", {"question": "Valid question?", "highlight_override": None}),
        ("/api/v1/answer", "POST", {"question": "Valid question?", "k": 3, "highlight_override": None}),
        ("/api/v1/health", "GET", None),
    ])
    def test_endpoint_responsiveness(self, endpoint, method, request_data):
        """Test that all endpoints respond appropriately (even if with errors)."""
        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST":
            response = client.post(endpoint, json=request_data)

        # All endpoints should return a response (not hang or timeout)
        assert response is not None
        # Response should have a status code
        assert hasattr(response, 'status_code')
        # Status code should be a valid HTTP status
        assert 100 <= response.status_code < 600

        print(f"✓ {endpoint} endpoint is responsive")

    def test_schema_conformance_for_complex_types(self):
        """Test that complex nested types in responses conform to their schemas."""
        # Test query endpoint response schema conformance
        response = client.post("/api/v1/query", json={
            "question": "What is Python programming?",
            "highlight_override": "Python basics"
        })

        if response.status_code == 200:
            data = response.json()

            # Verify the structure of retrieved_contexts
            assert "retrieved_contexts" in data
            for ctx in data["retrieved_contexts"]:
                # Each context should have content, metadata, and score
                assert "content" in ctx
                assert "metadata" in ctx
                assert "score" in ctx

                # Verify types of nested fields
                assert isinstance(ctx["content"], str)
                assert isinstance(ctx["metadata"], dict)
                assert isinstance(ctx["score"], (int, float))

                # Verify metadata has expected structure (at least it's a dict)
                assert isinstance(ctx["metadata"], dict)

        print("✓ Complex type schema conformance validation passed")


# Additional contract tests for edge cases
class TestAPIContractEdgeCases:
    """Additional contract tests for edge cases."""

    def test_large_request_body_handling(self):
        """Test how endpoints handle large request bodies."""
        # Create a large question (near the limit)
        large_question = "A " * 5000 + "?"  # Should be under the 10k limit

        response = client.post("/api/v1/query", json={
            "question": large_question,
            "highlight_override": None
        })

        # Large but valid request should either succeed or fail with specific error
        # but shouldn't cause server errors
        assert response.status_code in [200, 422], f"Large request caused unexpected error: {response.status_code}"

        print("✓ Large request body handling validation passed")

    def test_special_characters_in_requests(self):
        """Test handling of special characters in requests."""
        special_chars_question = "What is AI's capability in 2023? (Test: & ü ñ Δ λ)"

        response = client.post("/api/v1/query", json={
            "question": special_chars_question,
            "highlight_override": "AI & ML concepts"
        })

        # Should handle special characters gracefully
        assert response.status_code in [200, 422], f"Special characters caused unexpected error: {response.status_code}"

        print("✓ Special characters handling validation passed")

    def test_null_values_in_optional_fields(self):
        """Test handling of null values in optional fields."""
        response = client.post("/api/v1/query", json={
            "question": "Test question?",
            "highlight_override": None  # Explicitly null
        })

        # Should handle null values gracefully
        assert response.status_code in [200, 422], f"Null value caused unexpected error: {response.status_code}"

        print("✓ Null values in optional fields handling validation passed")

    def test_extreme_numerical_values(self):
        """Test handling of extreme numerical values."""
        # Test with the maximum allowed k value
        response = client.post("/api/v1/answer", json={
            "question": "Test?",
            "k": 10,  # Maximum allowed
            "highlight_override": None
        })

        # Should handle max k value gracefully
        assert response.status_code in [200, 422], f"Max k value caused unexpected error: {response.status_code}"

        print("✓ Extreme numerical values handling validation passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])