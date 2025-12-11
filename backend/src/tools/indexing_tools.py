from typing import Dict, List, Any
from ..agent.indexing_agent import IndexingAgent


class IndexingTools:
    """
    Tools for content indexing and management for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        """
        Initialize indexing tools with the indexing agent.
        """
        self.indexing_agent = IndexingAgent()

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
        return self.indexing_agent.index_content(content, source_file, document_type, section)

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
        return self.indexing_agent.update_content(content, source_file, document_type, section)

    def delete_content(self, source_file: str) -> Dict[str, Any]:
        """
        Delete content from the vector database by source file.

        Args:
            source_file: Name of the source file to delete

        Returns:
            Dictionary with deletion results
        """
        return self.indexing_agent.delete_content(source_file)

    def validate_content_format(self, content: str, document_type: str) -> Dict[str, Any]:
        """
        Validate that content is in the expected format.

        Args:
            content: Content to validate
            document_type: Expected document type ("markdown" or "html")

        Returns:
            Dictionary with validation results
        """
        return self.indexing_agent.validate_content_format(content, document_type)

    def get_indexing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexing system.

        Returns:
            Dictionary with indexing system statistics
        """
        return self.indexing_agent.get_indexing_stats()

    def batch_index_content(self, contents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Index multiple content items in a batch operation.

        Args:
            contents: List of content dictionaries with keys: content, source_file, document_type, section

        Returns:
            List of indexing results for each content item
        """
        results = []
        for item in contents:
            result = self.index_content(
                content=item['content'],
                source_file=item['source_file'],
                document_type=item.get('document_type', 'markdown'),
                section=item.get('section')
            )
            results.append(result)
        return results


# Example usage and tool definitions for agent integration
def create_indexing_tool_definitions() -> List[Dict]:
    """
    Create tool definitions for integration with OpenAI Agents API.

    Returns:
        List of tool definitions in OpenAI-compatible format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "index_content",
                "description": "Index content by chunking, embedding, and storing in vector database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to index"
                        },
                        "source_file": {
                            "type": "string",
                            "description": "Name of the source file"
                        },
                        "document_type": {
                            "type": "string",
                            "description": "Type of document (markdown or html), defaults to markdown",
                            "enum": ["markdown", "html"]
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional section identifier"
                        }
                    },
                    "required": ["content", "source_file"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_content",
                "description": "Update existing content by deleting old content and indexing new content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The updated content to index"
                        },
                        "source_file": {
                            "type": "string",
                            "description": "Name of the source file"
                        },
                        "document_type": {
                            "type": "string",
                            "description": "Type of document (markdown or html), defaults to markdown",
                            "enum": ["markdown", "html"]
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional section identifier"
                        }
                    },
                    "required": ["content", "source_file"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_content",
                "description": "Delete content from the vector database by source file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_file": {
                            "type": "string",
                            "description": "Name of the source file to delete"
                        }
                    },
                    "required": ["source_file"]
                }
            }
        }
    ]


if __name__ == "__main__":
    tools = IndexingTools()

    # Example of using the tools
    print("Indexing Tools initialized successfully")
    print(f"Available functions: index_content, update_content, delete_content, etc.")

    # Example tool definitions for agent integration
    tool_defs = create_indexing_tool_definitions()
    print(f"Generated {len(tool_defs)} tool definitions for agent integration")