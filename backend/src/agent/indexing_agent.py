from typing import Dict, List, Any
from ..rag.chunker import SemanticChunker, TextChunk
from ..rag.retriever import Retriever
from ..db.postgres_client import PostgresClient
import uuid


class IndexingAgent:
    """
    Specialized agent responsible for content ingestion in Markdown/HTML,
    semantic chunking, and vector database operations for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize the Indexing Agent with necessary components.
        """
        self.chunker = SemanticChunker()
        self.retriever = Retriever()
        self.postgres_client = PostgresClient()

    def index_content(self, content: str, source_file: str, document_type: str = "markdown",
                     section: str = None) -> Dict[str, Any]:
        """
        Index content by chunking, embedding, and storing in vector database.

        Args:
            content: The content to index
            source_file: Name of the source file
            document_type: Type of document ("markdown" or "html")
            section: Optional section identifier

        Returns:
            Dictionary with indexing results
        """
        start_time = 0
        import time
        start_time = time.time()

        try:
            # Chunk the content based on document type
            if document_type.lower() == "markdown":
                chunks = self.chunker.chunk_markdown(content, source_file, section)
            elif document_type.lower() == "html":
                chunks = self.chunker.chunk_html(content, source_file, section)
            else:
                # Default to markdown processing
                chunks = self.chunker.chunk_markdown(content, source_file, section)

            # Add chunks to the vector database
            success_count = self.retriever.batch_add_content(chunks)

            # Prepare result
            result = {
                "status": "success" if success_count == len(chunks) else "partial",
                "indexed_chunks": success_count,
                "total_chunks": len(chunks),
                "content_id": f"content_{uuid.uuid4().hex[:8]}",
                "processing_time": time.time() - start_time,
                "source_file": source_file
            }

            # Log the indexing operation
            self._log_indexing_operation(result)

            return result

        except Exception as e:
            result = {
                "status": "error",
                "error": str(e),
                "indexed_chunks": 0,
                "content_id": f"content_{uuid.uuid4().hex[:8]}",
                "processing_time": time.time() - start_time,
                "source_file": source_file
            }
            return result

    def update_content(self, content: str, source_file: str, document_type: str = "markdown",
                      section: str = None) -> Dict[str, Any]:
        """
        Update existing content by deleting old content and indexing new content.

        Args:
            content: The updated content to index
            source_file: Name of the source file
            document_type: Type of document ("markdown" or "html")
            section: Optional section identifier

        Returns:
            Dictionary with update results
        """
        import time
        start_time = time.time()

        try:
            # First, delete existing content from this source
            delete_success = self.retriever.delete_content_by_source(source_file)

            if not delete_success:
                print(f"Warning: Could not delete existing content for {source_file}")

            # Then index the new content
            index_result = self.index_content(content, source_file, document_type, section)

            # Update the result to reflect update operation
            result = {
                "status": index_result["status"],
                "indexed_chunks": index_result["indexed_chunks"],
                "total_chunks": index_result["total_chunks"],
                "content_id": index_result["content_id"],
                "processing_time": time.time() - start_time,
                "source_file": source_file,
                "update_operation": True
            }

            return result

        except Exception as e:
            result = {
                "status": "error",
                "error": str(e),
                "indexed_chunks": 0,
                "content_id": f"content_{uuid.uuid4().hex[:8]}",
                "processing_time": time.time() - start_time,
                "source_file": source_file,
                "update_operation": True
            }
            return result

    def delete_content(self, source_file: str) -> Dict[str, Any]:
        """
        Delete content from the vector database by source file.

        Args:
            source_file: Name of the source file to delete

        Returns:
            Dictionary with deletion results
        """
        import time
        start_time = time.time()

        try:
            # Delete from vector database
            delete_success = self.retriever.delete_content_by_source(source_file)

            if delete_success:
                result = {
                    "status": "success",
                    "deleted_source": source_file,
                    "processing_time": time.time() - start_time
                }
            else:
                result = {
                    "status": "error",
                    "error": "Failed to delete content from vector database",
                    "deleted_source": source_file,
                    "processing_time": time.time() - start_time
                }

            return result

        except Exception as e:
            result = {
                "status": "error",
                "error": str(e),
                "deleted_source": source_file,
                "processing_time": time.time() - start_time
            }
            return result

    def validate_content_format(self, content: str, document_type: str) -> Dict[str, Any]:
        """
        Validate that content is in the expected format.

        Args:
            content: Content to validate
            document_type: Expected document type ("markdown" or "html")

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        if not content or not content.strip():
            validation_result["is_valid"] = False
            validation_result["errors"].append("Content cannot be empty")

        if document_type.lower() not in ["markdown", "html"]:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Document type must be 'markdown' or 'html'")

        # Additional validation can be added here based on document type
        if document_type.lower() == "html":
            # Check for potentially dangerous HTML
            dangerous_patterns = ["<script", "javascript:", "vbscript:", "onerror", "onload"]
            for pattern in dangerous_patterns:
                if pattern.lower() in content.lower():
                    validation_result["warnings"].append(f"Potentially dangerous pattern found: {pattern}")

        return validation_result

    def _log_indexing_operation(self, result: Dict[str, Any]):
        """
        Log the indexing operation to the database.

        Args:
            result: Result of the indexing operation
        """
        try:
            # Log to the agent execution log
            self.postgres_client.log_agent_execution(
                agent_id="IndexingAgent",
                session_id=None,  # Indexing operations may not have a session
                tool_calls=["index_content"],
                input_params={"source_file": result.get("source_file")},
                output_result=str(result),
                execution_time=result.get("processing_time", 0)
            )
        except Exception as e:
            print(f"Error logging indexing operation: {e}")

    def get_indexing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexing system.

        Returns:
            Dictionary with indexing system statistics
        """
        retriever_stats = self.retriever.get_retrieval_stats()

        return {
            "retriever_stats": retriever_stats,
            "chunker_info": {
                "max_chunk_size": self.chunker.max_chunk_size,
                "min_chunk_size": self.chunker.min_chunk_size,
                "overlap": self.chunker.overlap
            }
        }


# Example usage
if __name__ == "__main__":
    agent = IndexingAgent()

    # Example content to index
    sample_markdown = """
# Introduction to Machine Learning

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

## Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
"""

    result = agent.index_content(sample_markdown, "ml_intro.md", "markdown", "Chapter 1")
    print(f"Indexing result: {result}")