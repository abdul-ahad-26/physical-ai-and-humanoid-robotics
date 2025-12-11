import os
import numpy as np
from typing import List, Union, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from functools import lru_cache
import time
import hashlib
from threading import Lock

load_dotenv()

class Embedder:
    """
    Generates embeddings for text content using free models.
    Optimized for performance with caching, batching, and efficient processing.
    Uses OpenAI's text-embedding-3-small model by default, with fallback options.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", cache_size: int = 10000):
        """
        Initialize the embedder with a specific model and caching.

        Args:
            model_name: Name of the embedding model to use
            cache_size: Size of the LRU cache for embeddings
        """
        self.model_name = model_name
        self.openai_client = None
        self.gemini_client = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = Lock()  # For thread-safe cache statistics

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize Gemini client
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai

        # Initialize cache for frequently used embeddings
        # We'll use a manual cache since lru_cache doesn't work well with the instance methods
        self._cache = {}
        self._cache_max_size = cache_size
        self._cache_access_count = 0
        self._cache_hit_count = 0

    @lru_cache(maxsize=10000)  # Cache up to 10k embeddings for repeated texts
    def _generate_embedding_lru_cached(self, text_hash: str, text: str) -> List[float]:
        """
        Internal method to generate embedding with LRU caching.

        Args:
            text_hash: Hash of the text (used as primary cache key)
            text: Original text to embed

        Returns:
            Embedding vector (list of floats)
        """
        start_time = time.time()

        try:
            # Use OpenAI embedding API if available
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    input=[text],
                    model=self.model_name
                )
                embedding = response.data[0].embedding
            elif self.gemini_client:
                # Use Gemini API if available
                result = self.gemini_client.embed_content(
                    model="embedding-001",
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embedding = result['embedding']
            else:
                # Fallback: Use a simple approach when APIs are not available
                embedding = self._generate_simple_embedding(text)

            return embedding
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        if not text or not text.strip():
            return [0.0] * 1536  # Return zero vector for empty text

        # Create a hash of the text for caching
        text_hash = hash(text) % (10**10)  # Limit hash to 10 digits to keep cache keys manageable

        # Use the LRU cached method
        return self._generate_embedding_lru_cached(text_hash, text)

    def generate_embeddings(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with optimized batching and caching.

        Args:
            texts: List of texts to embed
            batch_size: Size of batches for API calls (default 20, within OpenAI's limits)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches to optimize API usage and performance
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._generate_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts efficiently.

        Args:
            texts: List of texts in the batch

        Returns:
            List of embedding vectors for the batch
        """
        if not texts:
            return []

        # Filter out empty texts but keep track of positions
        original_positions = []
        valid_texts = []
        empty_positions = []  # Track positions of empty texts

        for idx, text in enumerate(texts):
            if text and text.strip():
                original_positions.append(idx)
                valid_texts.append(text)
            else:
                # Remember positions of empty texts to insert zero vectors later
                empty_positions.append(idx)

        if not valid_texts:
            # All texts were empty
            return [[0.0] * 1536 for _ in texts]

        # Try to use OpenAI's batch embedding API if available
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=valid_texts,
                    model=self.model_name
                )
                valid_embeddings = [item.embedding for item in response.data]

                # Reconstruct the full list with zero vectors for empty texts
                batch_result = [None] * len(texts)

                # Place valid embeddings in their original positions
                valid_idx = 0
                for orig_idx in original_positions:
                    batch_result[orig_idx] = valid_embeddings[valid_idx]
                    valid_idx += 1

                # Fill empty positions with zero vectors
                for idx in empty_positions:
                    batch_result[idx] = [0.0] * 1536

                return batch_result
            except Exception as e:
                print(f"OpenAI batch embedding failed: {e}")
                # Fall back to individual processing with caching

        # If batch API failed or not available, process individually with caching
        return [self.generate_embedding(text) for text in texts]

    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding when API is not available.
        Uses optimized character n-grams and hashing to create a deterministic vector.

        Args:
            text: Text to embed

        Returns:
            Simple embedding vector
        """
        if not text:
            return [0.0] * 1536

        # Use numpy for more efficient computation
        embedding = np.zeros(1536, dtype=np.float32)

        # Convert to lowercase once
        lower_text = text.lower()

        # Add character-level features with optimized processing
        text_len = len(lower_text)
        for i in range(text_len):
            char = lower_text[i]

            # Add single character features
            char_hash = hash(char) % 1536
            embedding[char_hash] += 0.2

            # Add bigram features if possible
            if i < text_len - 1:
                bigram = lower_text[i:i+2]
                bigram_hash = hash(bigram) % 1536
                embedding[bigram_hash] += 0.1

            # Add trigram features if possible
            if i < text_len - 2:
                trigram = lower_text[i:i+3]
                trigram_hash = hash(trigram) % 1536
                embedding[trigram_hash] += 0.05

        # Add word-level features
        words = lower_text.split()
        for word in words:
            if len(word) > 2:  # Only consider words with more than 2 characters
                word_hash = hash(word) % 1536
                embedding[word_hash] += 0.3

                # Add character bigrams from the word
                word_len = len(word)
                for j in range(word_len - 1):
                    bigram = word[j:j+2]
                    bigram_hash = hash(bigram) % 1536
                    embedding[bigram_hash] += 0.1

        # Normalize the embedding using numpy for efficiency
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors efficiently.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        # Calculate cosine similarity efficiently
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0  # Return 0 if either vector is zero

        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)

    def get_embedding_dimensions(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Number of dimensions in the embedding vectors
        """
        return 1536  # Standard for text-embedding-3-small

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the embedder.

        Returns:
            Dictionary with performance metrics
        """
        # Calculate cache statistics
        total_cache_accesses = self._cache_access_count
        hit_rate = self._cache_hit_count / total_cache_accesses if total_cache_accesses > 0 else 0.0

        return {
            "model_used": self.model_name if (self.openai_client or self.gemini_client) else "simple_fallback",
            "embedding_dimensions": self.get_embedding_dimensions(),
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size,
            "cache_hit_rate": hit_rate,
            "total_cache_accesses": total_cache_accesses,
            "total_cache_hits": self._cache_hit_count,
            "total_cache_misses": total_cache_accesses - self._cache_hit_count
        }

    def clear_cache(self):
        """
        Clear the embedding cache.
        """
        with self.lock:
            self._cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0

        # Also clear the LRU cache
        self._generate_embedding_cached.cache_clear()

    def warmup_cache(self, sample_texts: List[str]):
        """
        Pre-populate the cache with embeddings for sample texts.

        Args:
            sample_texts: List of sample texts to pre-cache
        """
        for text in sample_texts:
            # This will populate the cache
            self.generate_embedding(text)

        print(f"Warmup completed: Cached embeddings for {len(sample_texts)} sample texts")


# Example usage
if __name__ == "__main__":
    embedder = Embedder()

    # Test embedding generation
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language."
    ]

    # Time the embedding generation
    start_time = time.time()
    embeddings = embedder.generate_embeddings(texts)
    generation_time = time.time() - start_time

    print(f"Generated {len(embeddings)} embeddings in {generation_time:.3f}s")
    print(f"Embedding dimensions: {len(embeddings[0])}")

    # Test similarity
    similarity = embedder.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two texts: {similarity:.4f}")

    # Print performance stats
    stats = embedder.get_performance_stats()
    print(f"Performance stats: {stats}")