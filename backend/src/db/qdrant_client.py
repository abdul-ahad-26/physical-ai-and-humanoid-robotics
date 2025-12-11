import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

class QdrantClientWrapper:
    """
    Wrapper class for Qdrant client to handle vector database operations
    for the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")

        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)

        # Define collection name for textbook content
        self.collection_name = "textbook_content"

    def initialize_collection(self, vector_size: int = 1536):
        """
        Initialize the collection for storing textbook content embeddings.
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection '{self.collection_name}'")

    def upsert_vectors(self, vectors: List[Dict], metadata: List[Dict] = None):
        """
        Upsert vectors into the collection.

        Args:
            vectors: List of vectors to store
            metadata: List of metadata dictionaries corresponding to each vector
        """
        if metadata is None:
            metadata = [{}] * len(vectors)

        # Prepare points for upsert
        points = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            points.append(models.PointStruct(
                id=i,
                vector=vector,
                payload=meta
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_vectors(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for similar vectors in the collection.

        Args:
            query_vector: Vector to search for
            limit: Number of results to return

        Returns:
            List of search results with payload and score
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })

        return formatted_results

    def get_filter_by_source(self, source_file: str):
        """
        Create a filter for retrieving content by source file.

        Args:
            source_file: Name of the source file to filter by

        Returns:
            Filter object for Qdrant search
        """
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source_file",
                    match=models.MatchValue(value=source_file)
                )
            ]
        )

    def delete_by_payload(self, key: str, value: str):
        """
        Delete vectors by payload criteria.

        Args:
            key: Payload key to match
            value: Payload value to match
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    ]
                )
            )
        )

    def health_check(self) -> bool:
        """
        Check if Qdrant service is available.

        Returns:
            True if service is available, False otherwise
        """
        try:
            self.client.get_collection(self.collection_name)
            return True
        except:
            return False