from typing import Dict, List, Any
from ..agent.rag_agent import RAGAgent


class RAGTools:
    """
    Tools for RAG retrieval operations for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize RAG tools with the RAG Agent.
        """
        self.rag_agent = RAGAgent()

    def retrieve_context(self, query: str, k: int = 5, highlight_override: str = None) -> Dict[str, Any]:
        """
        Retrieve context for a given query.

        Args:
            query: The user's question or query
            k: Number of content chunks to retrieve
            highlight_override: Optional highlighted text that replaces search context

        Returns:
            Dictionary with retrieved contexts and assembled context
        """
        return self.rag_agent.process_query(query, k, highlight_override)

    def retrieve_by_source(self, source_file: str, k: int = 5) -> List[Dict]:
        """
        Retrieve content chunks from a specific source file.

        Args:
            source_file: Name of the source file to retrieve from
            k: Number of content chunks to retrieve

        Returns:
            List of content chunks from the specified source
        """
        return self.rag_agent.retrieve_by_source(source_file, k)

    def add_content_to_knowledge_base(self, content: str, metadata: Dict = None) -> bool:
        """
        Add content to the knowledge base.

        Args:
            content: Text content to add
            metadata: Additional metadata about the content (source_file, section, etc.)

        Returns:
            True if content was added successfully, False otherwise
        """
        if metadata is None:
            metadata = {}
        return self.rag_agent.add_content(content, metadata)

    def batch_add_content_to_knowledge_base(self, contents: List[Dict]) -> int:
        """
        Add multiple content chunks to the knowledge base.

        Args:
            contents: List of content dictionaries with 'content' and 'metadata' keys

        Returns:
            Number of successfully added chunks
        """
        from ..rag.chunker import TextChunk

        # Convert the input format to TextChunk objects
        text_chunks = []
        for content_dict in contents:
            chunk = TextChunk(
                content=content_dict['content'],
                metadata=content_dict.get('metadata', {}),
                start_pos=content_dict.get('start_pos', 0),
                end_pos=content_dict.get('end_pos', len(content_dict['content']))
            )
            text_chunks.append(chunk)

        return self.rag_agent.batch_add_content(text_chunks)

    def delete_content_from_knowledge_base(self, source_file: str) -> bool:
        """
        Delete content from the knowledge base by source file.

        Args:
            source_file: Name of the source file to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        return self.rag_agent.delete_content_by_source(source_file)

    def rank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rank retrieved results based on relevance to the query.

        Args:
            query: The original query
            results: List of retrieved results

        Returns:
            Ranked list of results
        """
        # Simple ranking based on the score from the vector search
        # In a more sophisticated implementation, this could use additional
        # ranking algorithms like cross-encoders
        ranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        return ranked_results

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.

        Returns:
            Dictionary with retrieval system statistics
        """
        return self.rag_agent.get_retrieval_stats()


# Example usage and tool definitions for agent integration
def create_rag_tool_definitions() -> List[Dict]:
    """
    Create tool definitions for integration with OpenAI Agents API.

    Returns:
        List of tool definitions in OpenAI-compatible format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve_context",
                "description": "Retrieve context for a given query from the knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question or query"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of content chunks to retrieve (default 5)",
                            "default": 5
                        },
                        "highlight_override": {
                            "type": "string",
                            "description": "Optional highlighted text that replaces search context"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_content_to_knowledge_base",
                "description": "Add content to the knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Text content to add"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata about the content (source_file, section, etc.)"
                        }
                    },
                    "required": ["content"]
                }
            }
        }
    ]


if __name__ == "__main__":
    tools = RAGTools()

    # Example of using the tools
    print("RAG Tools initialized successfully")
    print(f"Available functions: retrieve_context, retrieve_by_source, add_content_to_knowledge_base, etc.")

    # Example tool definitions for agent integration
    tool_defs = create_rag_tool_definitions()
    print(f"Generated {len(tool_defs)} tool definitions for agent integration")