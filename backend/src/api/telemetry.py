from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Set up the tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter for traces (if configured)
otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4318")
otlp_headers = os.getenv("OTLP_HEADERS")  # Format: "key1=value1,key2=value2"

if otlp_headers:
    # Parse headers from the environment variable
    headers = {}
    for header in otlp_headers.split(","):
        key, value = header.split("=", 1)
        headers[key.strip()] = value.strip()
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, headers=headers)
else:
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)

# Add the exporter to the tracer provider
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Initialize instrumentors for different components
def setup_telemetry(app=None):
    """
    Set up OpenTelemetry instrumentation for the application.

    Args:
        app: FastAPI application instance (optional)
    """
    # Instrument FastAPI application if provided
    if app:
        FastAPIInstrumentor.instrument_app(app)

    # Instrument requests library
    RequestsInstrumentor().instrument()

    # Instrument system metrics
    SystemMetricsInstrumentor().instrument()

    print("OpenTelemetry instrumentation set up successfully")


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer with the specified name.

    Args:
        name: Name of the tracer

    Returns:
        Configured tracer instance
    """
    return trace.get_tracer(name)


def add_span_attributes(span: trace.Span, attributes: dict):
    """
    Add attributes to the current span.

    Args:
        span: The span to add attributes to
        attributes: Dictionary of attributes to add
    """
    for key, value in attributes.items():
        span.set_attribute(key, str(value) if value is not None else "null")


def create_span(operation_name: str, attributes: Optional[dict] = None):
    """
    Context manager for creating spans with attributes.

    Args:
        operation_name: Name of the operation being traced
        attributes: Optional attributes to add to the span

    Yields:
        The created span
    """
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            add_span_attributes(span, attributes)
        yield span


# Example usage in different components
def trace_agent_execution(agent_name: str, operation: str, input_data: dict, output_data: dict):
    """
    Trace agent execution with relevant attributes.

    Args:
        agent_name: Name of the agent being traced
        operation: Operation being performed
        input_data: Input data for the operation
        output_data: Output data from the operation
    """
    with create_span(f"{agent_name}.{operation}", {
        "agent.name": agent_name,
        "operation": operation,
        "input.size": len(str(input_data)) if input_data else 0,
        "output.size": len(str(output_data)) if output_data else 0
    }) as span:
        # Add more specific attributes based on the operation
        if input_data and isinstance(input_data, dict):
            for key, value in list(input_data.items())[:5]:  # Only first 5 keys to avoid too much data
                span.set_attribute(f"input.{key}", str(value)[:100])  # Limit value length


def trace_database_operation(operation: str, table: str, query: str = None):
    """
    Trace database operations.

    Args:
        operation: Type of database operation (SELECT, INSERT, UPDATE, DELETE)
        table: Name of the table being operated on
        query: The actual query being executed (optional)
    """
    attributes = {
        "db.operation": operation,
        "db.table": table
    }

    if query:
        attributes["db.query"] = query[:200]  # Limit query length

    with create_span(f"database.{operation}", attributes):
        # Operation will be executed in the calling code
        pass


def trace_retrieval_operation(query: str, k: int, results_count: int):
    """
    Trace retrieval operations.

    Args:
        query: The query being used for retrieval
        k: Number of results requested
        results_count: Number of results returned
    """
    with create_span("retrieval.query", {
        "query.length": len(query) if query else 0,
        "query.k": k,
        "results.count": results_count
    }) as span:
        if query:
            span.set_attribute("query.text", query[:100])  # Limit query text length


def trace_api_request(endpoint: str, method: str, user_id: str = None, response_status: int = None):
    """
    Trace API requests with relevant attributes.

    Args:
        endpoint: The API endpoint being called
        method: HTTP method (GET, POST, etc.)
        user_id: ID of the user making the request (optional)
        response_status: HTTP response status (optional)
    """
    attributes = {
        "http.method": method,
        "http.route": endpoint,
    }

    if user_id:
        attributes["user.id"] = user_id

    if response_status:
        attributes["http.status_code"] = response_status

    with create_span(f"api.{method}.{endpoint}", attributes):
        # Request will be processed in the calling code
        pass


# Initialize telemetry when module is imported
def initialize_telemetry(app=None):
    """
    Initialize OpenTelemetry for the application.

    Args:
        app: FastAPI application instance (optional)
    """
    setup_telemetry(app)


# Example usage
if __name__ == "__main__":
    # Example of how to use the tracing in different parts of the application
    print("Setting up OpenTelemetry...")

    # This would normally be called when initializing the application
    initialize_telemetry()

    print("OpenTelemetry setup complete")

    # Example of tracing an agent operation
    with create_span("example.operation", {"param1": "value1", "param2": 42}):
        print("Performing traced operation...")

    print("Tracing example completed")