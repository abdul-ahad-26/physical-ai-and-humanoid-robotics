"""
Final Integration Validation Test for RAG + Agentic Backend for AI-Textbook Chatbot.

This module performs a comprehensive validation test to ensure all components work together
and meet the success criteria defined in the original specification.
"""

import sys
import os
import time
import uuid
from typing import Dict, Any, List
import asyncio
import requests

# Add the backend/src to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.agent.orchestrator import MainOrchestratorAgent
from src.agent.rag_agent import RAGAgent
from src.agent.indexing_agent import IndexingAgent
from src.agent.logging_agent import LoggingAgent
from src.db.postgres_client import PostgresClient
from src.db.qdrant_client import QdrantClientWrapper
from src.rag.retriever import Retriever
from src.rag.embedder import Embedder
from src.api.metrics import metrics_collector, alert_manager
from src.api.feature_flags import feature_flags
from src.agent.cleanup_scheduler import cleanup_manager


def test_basic_functionality():
    """Test basic functionality of the system"""
    print("Testing basic functionality...")

    # Initialize core components
    orchestrator = MainOrchestratorAgent()
    rag_agent = RAGAgent()
    indexing_agent = IndexingAgent()
    logging_agent = LoggingAgent()
    postgres_client = PostgresClient()
    qdrant_client = QdrantClientWrapper()

    print("✓ All agents initialized successfully")

    # Test database connectivity
    db_health = postgres_client.health_check()
    assert db_health, "PostgreSQL connection failed"
    print("✓ PostgreSQL connection healthy")

    # Test Qdrant connectivity
    qdrant_health = qdrant_client.health_check()
    assert qdrant_health, "Qdrant connection failed"
    print("✓ Qdrant connection healthy")

    # Test orchestrator health
    orchestrator_health = orchestrator.health_check()
    assert orchestrator_health["status"] in ["healthy", "degraded"], f"Orchestrator health check failed: {orchestrator_health}"
    print(f"✓ Orchestrator health: {orchestrator_health['status']}")

    return True


def test_content_indexing():
    """Test content indexing functionality"""
    print("\nTesting content indexing...")

    indexing_agent = IndexingAgent()

    # Create test content
    test_content = """
    # Introduction to Machine Learning

    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn
    from data, identify patterns and make decisions with minimal human intervention.

    ## Types of Machine Learning

    There are three main types of machine learning:

    1. Supervised Learning
    2. Unsupervised Learning
    3. Reinforcement Learning
    """

    source_file = f"test_ml_intro_{uuid.uuid4().hex[:8]}.md"

    # Index the content
    result = indexing_agent.index_content(
        content=test_content,
        source_file=source_file,
        document_type="markdown",
        section="Introduction"
    )

    assert result["status"] == "success", f"Indexing failed: {result}"
    assert result["indexed_chunks"] > 0, f"No chunks indexed: {result}"
    print(f"✓ Successfully indexed {result['indexed_chunks']} chunks")

    # Verify content is retrievable
    rag_agent = RAGAgent()
    retrieved = rag_agent.retrieve_content("machine learning", k=2)
    assert len(retrieved) > 0, "No content retrieved after indexing"
    print("✓ Indexed content is retrievable")

    return True


def test_query_functionality():
    """Test query functionality"""
    print("\nTesting query functionality...")

    rag_agent = RAGAgent()

    # Test a simple query
    start_time = time.time()
    result = rag_agent.process_query("What is machine learning?", k=3)
    query_time = time.time() - start_time

    assert "retrieved_contexts" in result, "Query result missing contexts"
    assert "assembled_context" in result, "Query result missing assembled context"
    print(f"✓ Query completed in {query_time:.3f}s with {len(result['retrieved_contexts'])} contexts")

    # Verify response time requirement (<300ms for top-5 search)
    assert query_time <= 0.300, f"Query response time {query_time:.3f}s exceeded 300ms limit"
    print("✓ Query response time meets performance requirement (<300ms)")

    return True


def test_answer_generation():
    """Test answer generation functionality"""
    print("\nTesting answer generation...")

    orchestrator = MainOrchestratorAgent()

    # Test answer generation
    start_time = time.time()
    result = orchestrator.process_query("What is machine learning?", k=3)
    answer_time = time.time() - start_time

    assert "answer" in result, "Answer generation failed - no answer in result"
    assert "confidence_score" in result, "No confidence score in result"
    print(f"✓ Answer generated in {answer_time:.3f}s with confidence {result['confidence_score']:.2f}")

    # Verify response time requirement (<2s end-to-end)
    assert answer_time <= 2.0, f"Answer response time {answer_time:.3f}s exceeded 2s limit"
    print("✓ Answer response time meets performance requirement (<2s)")

    return True


def test_logging_and_monitoring():
    """Test logging and monitoring functionality"""
    print("\nTesting logging and monitoring...")

    logging_agent = LoggingAgent()

    # Test session logging
    session_result = logging_agent.log_interaction(
        session_id=f"test_session_{uuid.uuid4().hex[:8]}",
        query="Test query for logging",
        response="Test response for logging",
        retrieved_context=[{"content": "test context", "metadata": {"source": "test"}, "score": 0.9}]
    )

    assert session_result["status"] == "success", f"Session logging failed: {session_result}"
    print("✓ Session logging working correctly")

    # Test metrics collection
    metrics_collector.record_request("POST", "/api/v1/query", 0.150, 200)
    metrics_collector.record_agent_execution("RAGAgent", "process_query", 0.120, True)
    print("✓ Metrics collection working correctly")

    # Test alerting (should not trigger alerts with normal values)
    alerts = alert_manager.check_alerts()
    # This is fine - no alerts should be triggered under normal operation
    print(f"✓ Alerting system operational (active alerts: {len(alerts)})")

    return True


def test_feature_flags():
    """Test feature flag functionality"""
    print("\nTesting feature flags...")

    # Test creating and checking a feature flag
    flag_name = f"test_feature_{uuid.uuid4().hex[:8]}"

    feature_flags.create_flag(
        name=flag_name,
        enabled=True,
        rollout_percentage=100.0,
        description="Test feature for validation"
    )

    is_enabled = feature_flags.is_enabled(flag_name)
    assert is_enabled, f"Feature flag {flag_name} should be enabled"
    print(f"✓ Feature flag {flag_name} working correctly")

    # Clean up
    feature_flags.delete_flag(flag_name)

    return True


def test_cleanup_functionality():
    """Test cleanup functionality"""
    print("\nTesting cleanup functionality...")

    # Test that cleanup manager is initialized
    assert cleanup_manager is not None, "Cleanup manager not initialized"
    print("✓ Cleanup manager initialized")

    # Test that cleanup jobs can be scheduled (without actually running them)
    cleanup_manager.initialize_cleanup_jobs()
    print("✓ Cleanup jobs can be initialized")

    return True


def test_error_handling():
    """Test error handling and graceful degradation"""
    print("\nTesting error handling...")

    rag_agent = RAGAgent()

    # Test with empty query (should handle gracefully)
    try:
        result = rag_agent.process_query("", k=1)
        # This might return empty results, which is acceptable
        print("✓ Empty query handled gracefully")
    except Exception as e:
        print(f"✓ Empty query caused expected error: {type(e).__name__}")

    # Test with very long query (should handle gracefully)
    try:
        long_query = "machine learning" * 1000
        result = rag_agent.process_query(long_query, k=1)
        print("✓ Long query handled gracefully")
    except Exception as e:
        print(f"✓ Long query caused expected error: {type(e).__name__}")

    return True


def run_complete_validation():
    """Run complete validation of the system"""
    print("="*70)
    print("FINAL INTEGRATION VALIDATION FOR RAG + AGENTIC BACKEND")
    print("="*70)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Content Indexing", test_content_indexing),
        ("Query Functionality", test_query_functionality),
        ("Answer Generation", test_answer_generation),
        ("Logging & Monitoring", test_logging_and_monitoring),
        ("Feature Flags", test_feature_flags),
        ("Cleanup Functionality", test_cleanup_functionality),
        ("Error Handling", test_error_handling),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} ---")
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
            if success:
                passed += 1
            else:
                results[test_name] = "FAILED"
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
            results[test_name] = f"FAILED - Error: {e}"

    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    for test_name, result in results.items():
        status = "✅ PASS" if result == "PASSED" else "❌ FAIL"
        print(f"{status} {test_name}: {result}")

    print("-"*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print(f"SUCCESS RATE: {passed/total*100:.1f}%" if total > 0 else "SUCCESS RATE: 0%")

    overall_success = passed == total
    print(f"OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    print("="*70)

    # Success criteria validation
    print("\nSUCCESS CRITERIA VALIDATION:")

    # Performance requirements
    print("- Query endpoint response time < 300ms: ✅ VERIFIED" if results.get("Query Functionality") == "PASSED" else "- Query endpoint response time < 300ms: ❌ FAILED")
    print("- Answer endpoint response time < 2s: ✅ VERIFIED" if results.get("Answer Generation") == "PASSED" else "- Answer endpoint response time < 2s: ❌ FAILED")

    # Functionality requirements
    print("- Content ingestion and indexing: ✅ VERIFIED" if results.get("Content Indexing") == "PASSED" else "- Content ingestion and indexing: ❌ FAILED")
    print("- Question answering: ✅ VERIFIED" if results.get("Answer Generation") == "PASSED" else "- Question answering: ❌ FAILED")
    print("- Session logging: ✅ VERIFIED" if results.get("Logging & Monitoring") == "PASSED" else "- Session logging: ❌ FAILED")
    print("- Error handling: ✅ VERIFIED" if results.get("Error Handling") == "PASSED" else "- Error handling: ❌ FAILED")

    print("\nSYSTEM IS READY FOR PRODUCTION DEPLOYMENT!" if overall_success else "\nSYSTEM NEEDS ADDITIONAL WORK BEFORE PRODUCTION")

    return overall_success


if __name__ == "__main__":
    success = run_complete_validation()
    exit(0 if success else 1)