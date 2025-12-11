"""
Performance Validation Tests for RAG + Agentic Backend for AI-Textbook Chatbot.

This module validates that the system meets the performance success criteria:
- Query endpoint response time < 300ms for top-5 search
- Answer endpoint response time < 2s end-to-end
- 80% agent tool call efficiency target
- Adequate throughput under load
- Resource utilization within acceptable bounds
"""

import time
import asyncio
import pytest
import statistics
from typing import List, Dict, Any
import uuid
import random
import string
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import os

# Import system components
from src.agent.orchestrator import MainOrchestratorAgent
from src.agent.rag_agent import RAGAgent
from src.agent.indexing_agent import IndexingAgent
from src.db.qdrant_client import QdrantClientWrapper
from src.rag.retriever import Retriever
from src.api.metrics import metrics_collector


class PerformanceValidator:
    """Validates system performance against success criteria"""

    def __init__(self):
        self.orchestrator = MainOrchestratorAgent()
        self.rag_agent = RAGAgent()
        self.qdrant_client = QdrantClientWrapper()
        self.retriever = Retriever()

        # Performance thresholds from success criteria
        self.query_response_threshold = 0.300  # 300ms
        self.answer_response_threshold = 2.0   # 2s
        self.agent_efficiency_target = 0.80    # 80%
        self.top_k_for_query = 5

        # Test data
        self.sample_questions = [
            "What is machine learning?",
            "Explain neural networks",
            "What is backpropagation?",
            "Describe supervised learning",
            "How does deep learning work?",
            "What are the types of machine learning?",
            "Explain reinforcement learning",
            "What is a decision tree?",
            "How do support vector machines work?",
            "What is natural language processing?"
        ]

        self.sample_content = """
        # Machine Learning Fundamentals

        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn
        from data, identify patterns and make decisions with minimal human intervention.

        ## Types of Machine Learning

        There are three main types of machine learning:

        1. Supervised Learning: Algorithms learn from labeled training data
        2. Unsupervised Learning: Algorithms find patterns in unlabeled data
        3. Reinforcement Learning: Algorithms learn through trial and error

        Supervised learning uses labeled examples to train models, while unsupervised
        learning discovers hidden patterns in data without guidance. Reinforcement
        learning uses rewards and punishments to learn optimal behaviors.

        ## Neural Networks

        Neural networks are computing systems inspired by biological neural networks.
        They consist of interconnected nodes that process information using dynamic
        state responses to external inputs.

        ### Backpropagation

        Backpropagation is the key algorithm that makes neural networks trainable.
        It calculates the gradient of the loss function with respect to the weights
        and propagates errors backward through the network.
        """

    def validate_query_performance(self) -> Dict[str, Any]:
        """Validate query endpoint performance (<300ms for top-5 search)"""
        print("Validating query performance...")

        response_times = []
        successful_queries = 0
        total_queries = len(self.sample_questions)

        # First, ensure content is available for querying
        self._ensure_test_content_available()

        for i, question in enumerate(self.sample_questions):
            start_time = time.time()
            try:
                result = self.rag_agent.process_query(
                    query=question,
                    k=self.top_k_for_query,
                    highlight_override=None
                )

                response_time = time.time() - start_time
                response_times.append(response_time)

                if response_time <= self.query_response_threshold:
                    successful_queries += 1
                else:
                    print(f"  Query {i+1} exceeded threshold: {response_time:.3f}s")

            except Exception as e:
                print(f"  Query {i+1} failed: {e}")

        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = self._calculate_percentile(response_times, 95) if response_times else 0
        p99_response_time = self._calculate_percentile(response_times, 99) if response_times else 0
        success_rate = successful_queries / total_queries if total_queries > 0 else 0

        results = {
            "metric": "query_response_time",
            "target": f"< {self.query_response_threshold}s",
            "actual_avg": f"{avg_response_time:.3f}s",
            "actual_p95": f"{p95_response_time:.3f}s",
            "actual_p99": f"{p99_response_time:.3f}s",
            "success_rate": f"{success_rate:.2%}",
            "passed": avg_response_time <= self.query_response_threshold and success_rate >= 0.95,
            "response_times": response_times,
            "summary": f"Query performance: Avg={avg_response_time:.3f}s, Success Rate={success_rate:.2%}"
        }

        print(f"  ✓ {results['summary']}")
        return results

    def validate_answer_performance(self) -> Dict[str, Any]:
        """Validate answer endpoint performance (<2s end-to-end)"""
        print("Validating answer performance...")

        response_times = []
        successful_answers = 0
        total_answers = len(self.sample_questions)

        for i, question in enumerate(self.sample_questions):
            start_time = time.time()
            try:
                result = self.orchestrator.process_query(
                    query=question,
                    k=3,  # Default k for answer endpoint
                    highlight_override=None
                )

                response_time = time.time() - start_time
                response_times.append(response_time)

                if response_time <= self.answer_response_threshold:
                    successful_answers += 1
                else:
                    print(f"  Answer {i+1} exceeded threshold: {response_time:.3f}s")

            except Exception as e:
                print(f"  Answer {i+1} failed: {e}")

        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = self._calculate_percentile(response_times, 95) if response_times else 0
        p99_response_time = self._calculate_percentile(response_times, 99) if response_times else 0
        success_rate = successful_answers / total_answers if total_answers > 0 else 0

        results = {
            "metric": "answer_response_time",
            "target": f"< {self.answer_response_threshold}s",
            "actual_avg": f"{avg_response_time:.3f}s",
            "actual_p95": f"{p95_response_time:.3f}s",
            "actual_p99": f"{p99_response_time:.3f}s",
            "success_rate": f"{success_rate:.2%}",
            "passed": avg_response_time <= self.answer_response_threshold and success_rate >= 0.95,
            "response_times": response_times,
            "summary": f"Answer performance: Avg={avg_response_time:.3f}s, Success Rate={success_rate:.2%}"
        }

        print(f"  ✓ {results['summary']}")
        return results

    def validate_agent_efficiency(self) -> Dict[str, Any]:
        """Validate agent tool call efficiency (80% target)"""
        print("Validating agent efficiency...")

        # Test the efficiency metrics from the orchestrator
        efficiency_metrics = self.orchestrator.get_execution_efficiency()

        actual_efficiency = efficiency_metrics.get("efficiency", 0)
        target_efficiency = self.agent_efficiency_target

        results = {
            "metric": "agent_tool_call_efficiency",
            "target": f">= {target_efficiency:.0%}",
            "actual": f"{actual_efficiency:.2%}",
            "total_executions": efficiency_metrics.get("total_executions", 0),
            "total_tool_calls": efficiency_metrics.get("total_tool_calls", 0),
            "successful_tool_calls": efficiency_metrics.get("successful_tool_calls", 0),
            "failed_tool_calls": efficiency_metrics.get("failed_tool_calls", 0),
            "passed": actual_efficiency >= target_efficiency,
            "summary": f"Agent efficiency: {actual_efficiency:.2%} (Target: {target_efficiency:.0%})"
        }

        print(f"  ✓ {results['summary']}")
        return results

    def validate_throughput_under_load(self) -> Dict[str, Any]:
        """Validate system throughput under load"""
        print("Validating throughput under load...")

        # Test concurrent query performance
        num_concurrent = 10
        queries_per_thread = 5
        total_queries = num_concurrent * queries_per_thread

        start_time = time.time()

        def worker(queries):
            thread_responses = []
            for i in range(queries_per_thread):
                question = random.choice(self.sample_questions)
                q_start = time.time()
                try:
                    result = self.rag_agent.process_query(
                        query=question,
                        k=3,
                        highlight_override=None
                    )
                    response_time = time.time() - q_start
                    thread_responses.append({"success": True, "time": response_time})
                except Exception as e:
                    thread_responses.append({"success": False, "time": time.time() - q_start, "error": str(e)})
            return thread_responses

        # Execute concurrently
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(worker, queries_per_thread) for _ in range(num_concurrent)]
            all_results = []
            for future in futures:
                all_results.extend(fast_future.result() for fast_future in futures)

        total_time = time.time() - start_time
        successful_queries = sum(1 for r in all_results if r["success"])
        response_times = [r["time"] for r in all_results if r["success"]]

        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        throughput = total_queries / total_time if total_time > 0 else 0

        results = {
            "metric": "concurrent_throughput",
            "target_queries": total_queries,
            "actual_queries": len(all_results),
            "successful_queries": successful_queries,
            "throughput_qps": f"{throughput:.2f}",
            "avg_response_time": f"{avg_response_time:.3f}s",
            "success_rate": f"{successful_queries/len(all_results):.2%}" if all_results else "0%",
            "total_time": f"{total_time:.2f}s",
            "passed": successful_queries/len(all_results) >= 0.90 if all_results else False,  # 90% success rate
            "summary": f"Throughput: {throughput:.2f} QPS, Success Rate: {successful_queries/len(all_results):.2%}" if all_results else "No results"
        }

        print(f"  ✓ {results['summary']}")
        return results

    def validate_resource_utilization(self) -> Dict[str, Any]:
        """Validate resource utilization under normal operation"""
        print("Validating resource utilization...")

        # Get current process info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        # Get system info
        system_memory = psutil.virtual_memory()
        system_cpu_percent = psutil.cpu_percent()

        # Get Qdrant resource usage if available
        qdrant_memory_mb = 0
        try:
            # This is a simplified check - in reality you'd need to check Qdrant process
            pass
        except:
            pass

        results = {
            "metric": "resource_utilization",
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "process_cpu_percent": cpu_percent,
            "system_memory_percent": system_memory.percent,
            "system_cpu_percent": system_cpu_percent,
            "passed": system_memory.percent < 80 and system_cpu_percent < 80,  # Reasonable limits
            "summary": f"Memory: {system_memory.percent}%, CPU: {system_cpu_percent}%"
        }

        print(f"  ✓ {results['summary']}")
        return results

    def validate_scalability_metrics(self) -> Dict[str, Any]:
        """Validate scalability-related performance metrics"""
        print("Validating scalability metrics...")

        # Test response time degradation under increasing load
        loads = [1, 5, 10, 20]  # Number of concurrent requests
        performance_degradation = []

        for load in loads:
            response_times = self._measure_response_time_at_load(load)
            avg_time = statistics.mean(response_times) if response_times else float('inf')
            performance_degradation.append({
                "load": load,
                "avg_response_time": avg_time,
                "num_successful": len(response_times)
            })

        # Calculate scalability coefficient (how much performance degrades with load)
        if len(performance_degradation) > 1:
            baseline_time = performance_degradation[0]["avg_response_time"]
            max_loaded_time = max(p["avg_response_time"] for p in performance_degradation)
            degradation_factor = max_loaded_time / baseline_time if baseline_time > 0 else float('inf')
        else:
            degradation_factor = 1.0

        results = {
            "metric": "scalability",
            "degradation_factor": f"{degradation_factor:.2f}x",
            "performance_data": performance_degradation,
            "passed": degradation_factor <= 3.0,  # Performance shouldn't degrade more than 3x
            "summary": f"Scalability factor: {degradation_factor:.2f}x (lower is better)"
        }

        print(f"  ✓ {results['summary']}")
        return results

    def run_complete_performance_validation(self) -> Dict[str, Any]:
        """Run complete performance validation suite"""
        print("="*60)
        print("RUNNING PERFORMANCE VALIDATION SUITE")
        print("="*60)

        results = {}

        # Run all validation tests
        results["query_performance"] = self.validate_query_performance()
        results["answer_performance"] = self.validate_answer_performance()
        results["agent_efficiency"] = self.validate_agent_efficiency()
        results["throughput"] = self.validate_throughput_under_load()
        results["resource_utilization"] = self.validate_resource_utilization()
        results["scalability"] = self.validate_scalability_metrics()

        # Overall summary
        all_passed = all(test["passed"] for test in results.values())

        summary = {
            "overall_result": "PASSED" if all_passed else "FAILED",
            "tests_run": len(results),
            "tests_passed": sum(1 for test in results.values() if test["passed"]),
            "tests_failed": sum(1 for test in results.values() if not test["passed"]),
            "detailed_results": results
        }

        print("\n" + "="*60)
        print("PERFORMANCE VALIDATION RESULTS")
        print("="*60)

        for test_name, result in results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}: {result['summary']}")

        print("-"*60)
        print(f"OVERALL: {summary['overall_result']}")
        print(f"Tests: {summary['tests_passed']}/{summary['tests_run']} passed")
        print("="*60)

        return summary

    def _ensure_test_content_available(self):
        """Ensure test content is available in the system"""
        try:
            # Try to retrieve something to see if content exists
            test_retrieval = self.retriever.retrieve_relevant_content("machine learning", k=1)
            if not test_retrieval:
                # Content doesn't exist, so add it
                indexing_agent = IndexingAgent()
                result = indexing_agent.index_content(
                    content=self.sample_content,
                    source_file=f"perf_test_content_{uuid.uuid4().hex[:8]}.md",
                    metadata={"source": "performance_test", "section": "ML_Fundamentals"}
                )
                print(f"  Added test content for performance validation: {result}")
        except Exception as e:
            print(f"  Warning: Could not ensure test content: {e}")

    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of response times"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)] if sorted_data else 0.0

    def _measure_response_time_at_load(self, concurrent_requests: int) -> List[float]:
        """Measure response times at a specific load level"""
        def single_request():
            start = time.time()
            try:
                question = random.choice(self.sample_questions)
                self.rag_agent.process_query(query=question, k=3)
                return time.time() - start
            except:
                return float('inf')  # Failed request

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(single_request) for _ in range(concurrent_requests)]
            response_times = []
            for future in futures:
                try:
                    time_taken = future.result(timeout=10.0)  # 10s timeout per request
                    if time_taken != float('inf'):
                        response_times.append(time_taken)
                except:
                    pass  # Request timed out

        return response_times


class PerformanceBenchmarkRunner:
    """Runs performance benchmarks and compares against success criteria"""

    def __init__(self):
        self.validator = PerformanceValidator()

    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("Starting performance benchmarking...")

        # Collect baseline metrics
        baseline_start_time = time.time()

        # Run the complete validation suite
        results = self.validator.run_complete_performance_validation()

        baseline_end_time = time.time()

        # Add timing information
        results["benchmark_duration"] = baseline_end_time - baseline_start_time

        return results

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed performance report"""
        report_lines = [
            "# Performance Validation Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Result:** {results['overall_result']}",
            f"**Duration:** {results['benchmark_duration']:.2f}s",
            "",
            "## Test Results",
            ""
        ]

        for test_name, test_result in results['detailed_results'].items():
            status_emoji = "✅" if test_result['passed'] else "❌"
            report_lines.append(f"### {status_emoji} {test_name.replace('_', ' ').title()}")
            report_lines.append(f"- **Result:** {'PASS' if test_result['passed'] else 'FAIL'}")
            report_lines.append(f"- **Details:** {test_result['summary']}")

            # Add specific metrics for each test
            if test_name == "query_performance":
                report_lines.append(f"- **Target:** < {self.validator.query_response_threshold}s")
                report_lines.append(f"- **Actual Average:** {test_result['actual_avg']}")
            elif test_name == "answer_performance":
                report_lines.append(f"- **Target:** < {self.validator.answer_response_threshold}s")
                report_lines.append(f"- **Actual Average:** {test_result['actual_avg']}")
            elif test_name == "agent_efficiency":
                report_lines.append(f"- **Target:** >= {self.validator.agent_efficiency_target:.0%}")
                report_lines.append(f"- **Actual:** {test_result['actual']}")

            report_lines.append("")

        report_lines.extend([
            "## Summary",
            f"- **Tests Passed:** {results['tests_passed']}/{results['tests_run']}",
            f"- **Success Rate:** {results['tests_passed']/results['tests_run']*100:.1f}%" if results['tests_run'] > 0 else "- **Success Rate:** 0%",
            "",
            "## Recommendations",
            "Based on the performance validation:"
        ])

        # Add recommendations based on results
        for test_name, test_result in results['detailed_results'].items():
            if not test_result['passed']:
                if test_name == "query_performance":
                    report_lines.append("- Consider optimizing vector search or adding caching for frequent queries")
                elif test_name == "answer_performance":
                    report_lines.append("- Review agent orchestration or implement more aggressive timeouts")
                elif test_name == "agent_efficiency":
                    report_lines.append("- Investigate agent tool call patterns and optimize inefficient operations")
                elif test_name == "throughput":
                    report_lines.append("- Consider scaling infrastructure or optimizing resource usage")

        if results['overall_result'] == "PASSED":
            report_lines.append("- Performance goals have been achieved")
        else:
            report_lines.append("- Performance improvements needed to meet success criteria")

        return "\n".join(report_lines)


def run_performance_validation():
    """Run the complete performance validation suite"""
    print("Starting Performance Validation for RAG + Agentic Backend")
    print("Success Criteria:")
    print("- Query endpoint response time < 300ms for top-5 search")
    print("- Answer endpoint response time < 2s end-to-end")
    print("- 80% agent tool call efficiency target")
    print("- Adequate throughput and resource utilization")
    print("")

    runner = PerformanceBenchmarkRunner()
    results = runner.run_benchmarks()

    # Generate and save report
    report = runner.generate_performance_report(results)

    # Save report to file
    report_filename = f"performance_validation_report_{int(time.time())}.md"
    with open(report_filename, 'w') as f:
        f.write(report)

    print(f"\nDetailed performance report saved to: {report_filename}")

    return results


if __name__ == "__main__":
    # Run performance validation
    results = run_performance_validation()

    # Exit with appropriate code
    if results['overall_result'] == 'PASSED':
        print("\n🎉 Performance validation PASSED - All success criteria met!")
        exit(0)
    else:
        print("\n❌ Performance validation FAILED - Some success criteria not met!")
        exit(1)