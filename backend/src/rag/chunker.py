import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata
    """
    content: str
    metadata: Dict
    start_pos: int
    end_pos: int


class SemanticChunker:
    """
    Implements semantic/content-aware chunking that respects document structure
    for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100, overlap: int = 100):
        """
        Initialize the semantic chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk_markdown(self, content: str, source_file: str = None, section: str = None) -> List[TextChunk]:
        """
        Chunk markdown content while respecting document structure.

        Args:
            content: The markdown content to chunk
            source_file: Name of the source file
            section: Section identifier

        Returns:
            List of TextChunk objects
        """
        # First, try to split by major document sections (headers)
        header_splits = self._split_by_headers(content)

        chunks = []
        for part in header_splits:
            part_chunks = self._chunk_by_size(part['content'],
                                            {'source_file': source_file, 'section': part['section'] or section})
            chunks.extend(part_chunks)

        return chunks

    def chunk_html(self, content: str, source_file: str = None, section: str = None) -> List[TextChunk]:
        """
        Chunk HTML content while respecting document structure.

        Args:
            content: The HTML content to chunk
            source_file: Name of the source file
            section: Section identifier

        Returns:
            List of TextChunk objects
        """
        # First, try to split by HTML sections (headers, divs, etc.)
        tag_splits = self._split_by_html_tags(content)

        chunks = []
        for part in tag_splits:
            part_chunks = self._chunk_by_size(part['content'],
                                            {'source_file': source_file, 'section': part['section'] or section})
            chunks.extend(part_chunks)

        return chunks

    def _split_by_headers(self, content: str) -> List[Dict]:
        """
        Split content by markdown headers to maintain document structure.
        """
        # Pattern to match markdown headers (h1, h2, h3, etc.)
        header_pattern = r'^(#{1,6})\s+(.*?)\n'

        # Split content by headers
        parts = []
        current_section = ""
        current_content = ""

        lines = content.split('\n')
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Save the previous section if it exists
                if current_content.strip():
                    parts.append({'content': current_content.strip(), 'section': current_section})

                # Start a new section with the header
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                current_section = f"{'#' * header_level} {header_text}"
                current_content = line + '\n'
            else:
                current_content += line + '\n'

        # Add the last section
        if current_content.strip():
            parts.append({'content': current_content.strip(), 'section': current_section})

        # If no headers were found, return the whole content as one part
        if not parts:
            parts.append({'content': content, 'section': ''})

        return parts

    def _split_by_html_tags(self, content: str) -> List[Dict]:
        """
        Split content by HTML section tags to maintain document structure.
        """
        # Pattern to match section tags like h1, h2, h3, div, section, article, etc.
        section_pattern = r'<(h[1-6]|div|section|article|p)[^>]*>.*?</\1>'

        # For now, we'll use a simpler approach that looks for header tags specifically
        # Find all header tags and their content
        header_pattern = r'(<h[1-6][^>]*>.*?</h[1-6]>)(.*?)(?=<h[1-6]|$)'
        matches = re.findall(header_pattern, content, re.DOTALL | re.IGNORECASE)

        parts = []
        if matches:
            for header_tag, content_part in matches:
                # Extract header text for the section name
                header_text = re.sub(r'<[^>]+>', '', header_tag).strip()
                parts.append({
                    'content': f"{header_tag}\n{content_part}".strip(),
                    'section': header_text
                })
        else:
            # If no headers found, return the whole content as one part
            parts.append({'content': content, 'section': ''})

        return parts

    def _chunk_by_size(self, content: str, metadata: Dict) -> List[TextChunk]:
        """
        Split content into chunks by size, with overlap and respecting sentence boundaries.
        """
        if len(content) <= self.max_chunk_size:
            # If content is small enough, return as a single chunk
            return [TextChunk(content=content, metadata=metadata, start_pos=0, end_pos=len(content))]

        chunks = []
        start = 0

        while start < len(content):
            end = start + self.max_chunk_size

            # If this is the last chunk and it's already small enough, just add it
            if end >= len(content) and (end - start) >= self.min_chunk_size:
                chunk_content = content[start:end]
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=metadata,
                    start_pos=start,
                    end_pos=end
                ))
                break

            # Try to split at sentence boundary
            if end < len(content):
                # Look for sentence boundaries near the end
                sentence_end = self._find_sentence_boundary(content, start, end)

                if sentence_end > start:
                    end = sentence_end
                else:
                    # If no sentence boundary found, try to split at word boundary
                    word_end = self._find_word_boundary(content, start, end)
                    if word_end > start:
                        end = word_end
                    else:
                        # If no word boundary found, just split at max size
                        end = min(end, len(content))

            # Extract the chunk content
            chunk_content = content[start:end]

            # Create the chunk with metadata
            chunk_metadata = metadata.copy()
            chunk_metadata['start_pos'] = start
            chunk_metadata['end_pos'] = end

            chunks.append(TextChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                start_pos=start,
                end_pos=end
            ))

            # Move start position forward with overlap
            start = end - self.overlap if self.overlap < end else end

            # If we're near the end, break
            if start >= len(content):
                break

        return chunks

    def _find_sentence_boundary(self, content: str, start: int, end: int) -> int:
        """
        Find the best sentence boundary near the end position.
        """
        # Look backwards from the end position for sentence endings
        for i in range(min(end, len(content)) - 1, max(start, end - 200), -1):
            if content[i] in '.!?。！？':
                # Check if the next character is a space or end of string
                if i + 1 >= len(content) or content[i + 1].isspace():
                    return i + 1

        return -1  # No sentence boundary found

    def _find_word_boundary(self, content: str, start: int, end: int) -> int:
        """
        Find the best word boundary near the end position.
        """
        # Look backwards from the end position for word boundaries
        for i in range(min(end, len(content)) - 1, max(start, end - 100), -1):
            if content[i].isspace():
                return i

        return -1  # No word boundary found


# Example usage
if __name__ == "__main__":
    chunker = SemanticChunker()

    # Example markdown content
    md_content = """
# Introduction to Machine Learning

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

## Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning uses rewards and penalties to learn optimal behaviors.
"""

    chunks = chunker.chunk_markdown(md_content, source_file="ml_intro.md", section="Introduction")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Section: {chunk.metadata.get('section', 'N/A')}")
        print(f"Content preview: {chunk.content[:100]}...")
        print(f"Length: {len(chunk.content)} characters\n")