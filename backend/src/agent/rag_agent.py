from typing import Dict, List, Any
from ..rag.retriever import Retriever
from ..rag.chunker import TextChunk
from ..db.models import UserSession, QueryContext


class RAGAgent:
    """
    Specialized agent responsible for retrieval-augmented generation tasks
    and content retrieval using semantic chunking for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize the RAG Agent with necessary components.
        """
        self.retriever = Retriever()

    def retrieve_content(self, query: str, k: int = 5, highlight_override: str = None) -> List[Dict]:
        """
        Retrieve relevant content based on the query.

        Args:
            query: The user's question or query
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text that replaces search context

        Returns:
            List of relevant content chunks with metadata and scores
        """
        return self.retriever.retrieve_with_fallback(query, k, highlight_override)

    def retrieve_by_source(self, source_file: str, k: int = 5) -> List[Dict]:
        """
        Retrieve content chunks from a specific source file.

        Args:
            source_file: Name of the source file to retrieve from
            k: Number of content chunks to retrieve

        Returns:
            List of content chunks from the specified source
        """
        return self.retriever.retrieve_by_source(source_file, k)

    def add_content(self, content: str, metadata: Dict = None) -> bool:
        """
        Add content to the vector database.

        Args:
            content: Text content to add
            metadata: Additional metadata about the content

        Returns:
            True if content was added successfully, False otherwise
        """
        return self.retriever.add_content(content, metadata)

    def batch_add_content(self, contents: List[TextChunk]) -> int:
        """
        Add multiple content chunks to the vector database.

        Args:
            contents: List of TextChunk objects to add

        Returns:
            Number of successfully added chunks
        """
        return self.retriever.batch_add_content(contents)

    def delete_content_by_source(self, source_file: str) -> bool:
        """
        Delete content from the vector database by source file.

        Args:
            source_file: Name of the source file to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        return self.retriever.delete_content_by_source(source_file)

    def process_query(self, query: str, k: int = 5, highlight_override: str = None) -> Dict[str, Any]:
        """
        Process a query by retrieving relevant content and preparing context.

        Args:
            query: The user's question
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text that replaces search context

        Returns:
            Dictionary with retrieved context and assembled prompt context
        """
        # Retrieve relevant content
        retrieved_chunks = self.retrieve_content(query, k, highlight_override)

        # Assemble the context for the LLM
        assembled_context = self._assemble_context(query, retrieved_chunks, highlight_override)

        return {
            "retrieved_contexts": retrieved_chunks,
            "assembled_context": assembled_context,
            "query_id": self._generate_query_id()
        }

    def _assemble_context(self, query: str, retrieved_chunks: List[Dict], highlight_override: str = None) -> str:
        """
        Assemble the context from retrieved chunks for the LLM.

        Args:
            query: Original user query
            retrieved_chunks: List of retrieved content chunks
            highlight_override: Optional highlighted text

        Returns:
            Assembled context string for the LLM
        """
        context_parts = []

        # Add highlight override if provided
        if highlight_override:
            context_parts.append(f"Highlighted context provided by user: {highlight_override}\n")

        # Add retrieved content chunks
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"Context {i+1} (from {chunk['metadata'].get('source_file', 'unknown')}):")
            context_parts.append(f"{chunk['content']}")
            context_parts.append("---")

        # Add the original question
        context_parts.append(f"User's question: {query}")

        return "\n".join(context_parts)

    def _generate_query_id(self) -> str:
        """
        Generate a unique query ID.

        Returns:
            Unique query identifier
        """
        import uuid
        return f"query_{uuid.uuid4().hex[:8]}"

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.

        Returns:
            Dictionary with retrieval system statistics
        """
        return self.retriever.get_retrieval_stats()


# Example usage
if __name__ == "__main__":
    agent = RAGAgent()

    # Example of processing a query
    result = agent.process_query("What is machine learning?", k=3)
    print(f"Retrieved {len(result['retrieved_contexts'])} contexts")
    print(f"Assembled context preview: {result['assembled_context'][:200]}...")