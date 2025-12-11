from typing import List, Dict, Optional
from .embedder import Embedder
from ..db.qdrant_client import QdrantClientWrapper
from ..db.models import TextbookContent
from .chunker import TextChunk
import json


class Retriever:
    """
    Handles retrieval of relevant content from the vector database
    for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize the retriever with embedder and Qdrant client.
        """
        self.embedder = Embedder()
        self.qdrant_client = QdrantClientWrapper()
        self.qdrant_client.initialize_collection(vector_size=self.embedder.get_embedding_dimensions())

    def retrieve_relevant_content(self, query: str, k: int = 5, highlight_override: str = None) -> List[Dict]:
        """
        Retrieve relevant content chunks based on the query.

        Args:
            query: The user's question or query
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text that replaces search context

        Returns:
            List of relevant content chunks with metadata and scores
        """
        try:
            # Use highlight override if provided, otherwise use the query
            search_text = highlight_override if highlight_override else query

            # Generate embedding for the search text
            query_embedding = self.embedder.generate_embedding(search_text)

            # Search in Qdrant
            search_results = self.qdrant_client.search_vectors(query_embedding, limit=k)

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "id": result["id"],
                    "content": result["payload"].get("content", ""),
                    "metadata": result["payload"].get("metadata", {}),
                    "score": result["score"]
                })

            return formatted_results
        except Exception as e:
            print(f"Qdrant retrieval failed: {e}")
            # Return empty results with graceful degradation
            return []

    def retrieve_with_fallback(self, query: str, k: int = 5, highlight_override: str = None) -> List[Dict]:
        """
        Retrieve content with fallback options when Qdrant is unavailable.

        Args:
            query: The user's question or query
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text that replaces search context

        Returns:
            List of relevant content chunks with metadata and scores
        """
        # First, try normal retrieval
        results = self.retrieve_relevant_content(query, k, highlight_override)

        # If no results and Qdrant is down, try alternative approaches
        if not results and not self.qdrant_client.health_check():
            # Implement fallback logic here
            # For now, return empty results with a note about fallback
            print("Qdrant unavailable, using fallback approach (currently returning empty results)")
            return []

        return results

    def retrieve_by_source(self, source_file: str, k: int = 5) -> List[Dict]:
        """
        Retrieve content chunks from a specific source file.

        Args:
            source_file: Name of the source file to retrieve from
            k: Number of content chunks to retrieve

        Returns:
            List of content chunks from the specified source
        """
        try:
            # Use Qdrant's scroll functionality to retrieve by source with filter
            results = self.qdrant_client.client.scroll(
                collection_name=self.qdrant_client.collection_name,
                scroll_filter=self.qdrant_client.get_filter_by_source(source_file),
                limit=k,
                with_payload=True,
                with_vectors=False
            )

            # The scroll method returns (records, next_page_offset)
            records = results[0]

            formatted_results = []
            for record in records:
                formatted_results.append({
                    "id": record.id,
                    "content": record.payload.get("content", ""),
                    "metadata": record.payload.get("metadata", {}),
                    "score": 1.0  # Default score since scroll doesn't provide similarity scores
                })

            return formatted_results
        except Exception as e:
            print(f"Error retrieving content by source: {e}")
            # Return empty results with graceful degradation
            return []

    def add_content(self, content: str, metadata: Dict = None) -> bool:
        """
        Add content to the vector database.

        Args:
            content: Text content to add
            metadata: Additional metadata about the content

        Returns:
            True if content was added successfully, False otherwise
        """
        if not content.strip():
            return False

        if metadata is None:
            metadata = {}

        try:
            # Generate embedding for the content
            content_embedding = self.embedder.generate_embedding(content)

            # Add to Qdrant
            self.qdrant_client.upsert_vectors([content_embedding], [metadata])
            return True
        except Exception as e:
            print(f"Error adding content to vector database: {e}")
            return False

    def batch_add_content(self, contents: List[TextChunk]) -> int:
        """
        Add multiple content chunks to the vector database.

        Args:
            contents: List of TextChunk objects to add

        Returns:
            Number of successfully added chunks
        """
        if not contents:
            return 0

        try:
            # Generate embeddings for all contents
            embeddings = []
            metadata_list = []
            for chunk in contents:
                embedding = self.embedder.generate_embedding(chunk.content)
                embeddings.append(embedding)
                # Merge the chunk's metadata with additional info
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata['content'] = chunk.content
                metadata_list.append(chunk_metadata)

            # Add to Qdrant
            self.qdrant_client.upsert_vectors(embeddings, metadata_list)
            return len(contents)
        except Exception as e:
            print(f"Error adding content batch to vector database: {e}")
            return 0

    def delete_content_by_source(self, source_file: str) -> bool:
        """
        Delete content from the vector database by source file.

        Args:
            source_file: Name of the source file to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.qdrant_client.delete_by_payload("metadata.source_file", source_file)
            return True
        except Exception as e:
            print(f"Error deleting content from vector database: {e}")
            return False

    def get_retrieval_stats(self) -> Dict:
        """
        Get statistics about the retrieval system.

        Returns:
            Dictionary with retrieval system statistics
        """
        # In a real implementation, this would query Qdrant for collection stats
        # For now, we'll return placeholder values
        return {
            "total_chunks": 0,  # This would be retrieved from Qdrant
            "collections": [self.qdrant_client.collection_name],
            "embedding_model": self.embedder.model_name,
            "embedding_dimensions": self.embedder.get_embedding_dimensions()
        }


# Example usage
if __name__ == "__main__":
    from .chunker import TextChunk

    retriever = Retriever()

    # Example content to add
    sample_content = TextChunk(
        content="Machine learning is a method of data analysis that automates analytical model building.",
        metadata={"source_file": "ml_basics.md", "section": "Introduction", "page_number": 1}
    )

    # Add content to the database
    success = retriever.add_content(sample_content.content, sample_content.metadata)
    print(f"Content added successfully: {success}")

    # Retrieve relevant content
    results = retriever.retrieve_relevant_content("What is machine learning?", k=1)
    print(f"Retrieved {len(results)} results:")
    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['content'][:100]}...")
        print(f"Metadata: {result['metadata']}")
        print("---")