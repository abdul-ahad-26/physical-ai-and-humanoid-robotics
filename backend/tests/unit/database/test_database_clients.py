import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.db.postgres_client import PostgresClient
from src.db.qdrant_client import QdrantClientWrapper
from datetime import datetime, timedelta


class TestPostgresClient:
    """Unit tests for PostgresClient class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the environment variable to avoid needing a real connection
        with patch('os.getenv', return_value='postgresql://test:test@localhost/test'):
            self.postgres_client = PostgresClient()

    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_initialization_success(self, mock_pool):
        """Test successful initialization of PostgresClient."""
        # Mock the pool creation
        mock_pool_instance = Mock()
        mock_pool.return_value = mock_pool_instance

        with patch('os.getenv', return_value='postgresql://user:pass@localhost/db'):
            client = PostgresClient()

        assert client.connection_string == 'postgresql://user:pass@localhost/db'
        assert mock_pool.called

    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_initialization_missing_env_var(self, mock_pool):
        """Test initialization fails when NEON_DB_URL is missing."""
        with patch('os.getenv', return_value=None):
            with pytest.raises(ValueError, match="NEON_DB_URL environment variable is required"):
                PostgresClient()

    @patch.object(PostgresClient, 'get_connection')
    def test_log_user_session_success(self, mock_get_conn):
        """Test successful logging of a user session."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {'id': 'session_123'}

        mock_get_conn.return_value = mock_conn

        session_id = self.postgres_client.log_user_session(
            query="Test question?",
            response="Test answer.",
            user_id="user_456",
            retrieved_context=[{"content": "test context", "score": 0.8}],
            session_metadata={"highlight_override": "highlighted text"}
        )

        assert session_id == "session_123"
        mock_cursor.execute.assert_called_once()
        assert 'INSERT INTO user_sessions' in str(mock_cursor.execute.call_args[0][0])

    @patch.object(PostgresClient, 'get_connection')
    def test_get_user_session_success(self, mock_get_conn):
        """Test successful retrieval of a user session."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {
            'id': 'session_123',
            'user_id': 'user_456',
            'query': 'Test question?',
            'response': 'Test answer.',
            'retrieved_context': '[{"content": "test context", "score": 0.8}]',
            'timestamp': datetime.now(),
            'session_metadata': '{}'
        }

        mock_get_conn.return_value = mock_conn

        session = self.postgres_client.get_user_session('session_123')

        assert session is not None
        assert session['id'] == 'session_123'
        assert session['user_id'] == 'user_456'
        assert session['query'] == 'Test question?'

    @patch.object(PostgresClient, 'get_connection')
    def test_get_user_session_not_found(self, mock_get_conn):
        """Test retrieval of a non-existent user session."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        mock_get_conn.return_value = mock_conn

        session = self.postgres_client.get_user_session('nonexistent_session')

        assert session is None

    @patch.object(PostgresClient, 'get_connection')
    def test_log_query_context_success(self, mock_get_conn):
        """Test successful logging of query context."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {'id': 'context_789'}

        mock_get_conn.return_value = mock_conn

        context_id = self.postgres_client.log_query_context(
            original_question="Test question?",
            session_id="session_123",
            highlight_override="highlighted text",
            retrieved_chunks=[{"content": "test content", "score": 0.8}],
            processed_context="processed context"
        )

        assert context_id == "context_789"
        mock_cursor.execute.assert_called_once()
        assert 'INSERT INTO query_contexts' in str(mock_cursor.execute.call_args[0][0])

    @patch.object(PostgresClient, 'get_connection')
    def test_log_agent_execution_success(self, mock_get_conn):
        """Test successful logging of agent execution."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {'id': 'log_101'}

        mock_get_conn.return_value = mock_conn

        log_id = self.postgres_client.log_agent_execution(
            agent_id="TestAgent",
            session_id="session_123",
            tool_calls=["test_tool"],
            input_params={"param": "value"},
            output_result="result",
            execution_time=0.25
        )

        assert log_id == "log_101"
        mock_cursor.execute.assert_called_once()
        assert 'INSERT INTO agent_execution_logs' in str(mock_cursor.execute.call_args[0][0])

    @patch.object(PostgresClient, 'get_connection')
    def test_cleanup_expired_sessions_success(self, mock_get_conn):
        """Test successful cleanup of expired sessions."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.rowcount = 5  # 5 rows deleted

        mock_get_conn.return_value = mock_conn

        deleted_count = self.postgres_client.cleanup_expired_sessions(days=30)

        assert deleted_count == 5
        mock_cursor.execute.assert_called_once()
        assert 'DELETE FROM user_sessions' in str(mock_cursor.execute.call_args[0][0])

    @patch.object(PostgresClient, 'get_connection')
    def test_health_check_success(self, mock_get_conn):
        """Test successful health check."""
        # Mock the connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_get_conn.return_value = mock_conn

        health = self.postgres_client.health_check()

        assert health is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @patch.object(PostgresClient, 'get_connection')
    def test_health_check_failure(self, mock_get_conn):
        """Test health check failure."""
        # Mock the connection to raise an exception
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("Connection failed")

        mock_get_conn.return_value = mock_conn

        health = self.postgres_client.health_check()

        assert health is False

    def test_return_connection(self):
        """Test returning connection to pool."""
        # Create a mock pool and client
        mock_pool = Mock()
        self.postgres_client.connection_pool = mock_pool

        mock_conn = Mock()
        self.postgres_client.return_connection(mock_conn)

        mock_pool.putconn.assert_called_once_with(mock_conn)


class TestQdrantClientWrapper:
    """Unit tests for QdrantClientWrapper class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('os.getenv', return_value='http://localhost:6333'):
            self.qdrant_client = QdrantClientWrapper()

    @patch('qdrant_client.QdrantClient')
    def test_initialization_with_api_key(self, mock_qdrant_client_class):
        """Test successful initialization of QdrantClientWrapper with API key."""
        mock_client_instance = Mock()
        mock_qdrant_client_class.return_value = mock_client_instance

        with patch('os.getenv', side_effect=lambda x: 'test-api-key' if x == 'QDRANT_API_KEY' else 'http://localhost:6333'):
            client = QdrantClientWrapper()

        mock_qdrant_client_class.assert_called_once_with(url='http://localhost:6333', api_key='test-api-key')

    @patch('qdrant_client.QdrantClient')
    def test_initialization_without_api_key(self, mock_qdrant_client_class):
        """Test successful initialization of QdrantClientWrapper without API key."""
        mock_client_instance = Mock()
        mock_qdrant_client_class.return_value = mock_client_instance

        with patch('os.getenv', side_effect=lambda x: None if x == 'QDRANT_API_KEY' else 'http://localhost:6333'):
            client = QdrantClientWrapper()

        mock_qdrant_client_class.assert_called_once_with(url='http://localhost:6333')

    @patch.object(QdrantClientWrapper, 'client')
    def test_initialize_collection_new_collection(self, mock_client):
        """Test initialization of a new collection."""
        # Mock the get_collection to raise an exception (collection doesn't exist)
        mock_client.get_collection.side_effect = Exception("Collection not found")

        self.qdrant_client.initialize_collection(vector_size=1536)

        # Verify that create_collection was called
        mock_client.create_collection.assert_called_once()

    @patch.object(QdrantClientWrapper, 'client')
    def test_initialize_collection_existing_collection(self, mock_client):
        """Test initialization when collection already exists."""
        # Mock the get_collection to return successfully (collection exists)
        mock_client.get_collection.return_value = Mock()

        self.qdrant_client.initialize_collection(vector_size=1536)

        # Verify that create_collection was NOT called
        mock_client.create_collection.assert_not_called()

    @patch.object(QdrantClientWrapper, 'client')
    def test_upsert_vectors_success(self, mock_client):
        """Test successful upsert of vectors."""
        mock_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_metadata = [{"source": "doc1"}, {"source": "doc2"}]

        self.qdrant_client.upsert_vectors(mock_vectors, mock_metadata)

        # Verify upsert was called with correct parameters
        mock_client.upsert.assert_called_once()

    @patch.object(QdrantClientWrapper, 'client')
    def test_search_vectors_success(self, mock_client):
        """Test successful vector search."""
        # Mock search results
        mock_search_result = [
            Mock(id=1, score=0.9, payload={"content": "test content", "metadata": {"source": "doc1"}}),
            Mock(id=2, score=0.8, payload={"content": "other content", "metadata": {"source": "doc2"}})
        ]
        mock_client.search.return_value = mock_search_result

        results = self.qdrant_client.search_vectors([0.1, 0.2, 0.3], limit=2)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["score"] == 0.9
        assert results[0]["payload"]["content"] == "test content"

    @patch.object(QdrantClientWrapper, 'client')
    def test_delete_by_payload_success(self, mock_client):
        """Test successful deletion by payload."""
        self.qdrant_client.delete_by_payload("source_file", "test.md")

        # Verify delete was called with correct parameters
        mock_client.delete.assert_called_once()

    @patch.object(QdrantClientWrapper, 'client')
    def test_health_check_success(self, mock_client):
        """Test successful health check."""
        mock_client.get_collection.return_value = Mock()

        health = self.qdrant_client.health_check()

        assert health is True

    @patch.object(QdrantClientWrapper, 'client')
    def test_health_check_failure(self, mock_client):
        """Test health check failure."""
        mock_client.get_collection.side_effect = Exception("Connection failed")

        health = self.qdrant_client.health_check()

        assert health is False


class TestDatabaseIntegration:
    """Integration tests for database clients working together."""

    @patch('psycopg2.pool.ThreadedConnectionPool')
    @patch('qdrant_client.QdrantClient')
    def test_session_logging_with_context_linkage(self, mock_qdrant_client, mock_pool):
        """Test that user sessions and query contexts are properly linked."""
        # Setup mocks
        mock_postgres_conn = Mock()
        mock_postgres_cursor = Mock()
        mock_postgres_conn.cursor.return_value.__enter__.return_value = mock_postgres_cursor
        mock_postgres_cursor.fetchone.return_value = {'id': 'session_123'}

        mock_pool_instance = Mock()
        mock_pool.return_value = mock_pool_instance
        mock_pool_instance.getconn.return_value = mock_postgres_conn

        mock_qdrant_instance = Mock()
        mock_qdrant_client.return_value = mock_qdrant_instance

        # Initialize clients with mocked dependencies
        with patch('os.getenv', side_effect=lambda x: 'postgresql://test@test/test' if 'NEON' in x else 'http://localhost:6333'):
            postgres_client = PostgresClient()

        # Log a session
        session_id = postgres_client.log_user_session(
            query="Test question?",
            response="Test answer.",
            user_id="user_456"
        )

        # Verify session was logged with correct ID
        assert session_id == "session_123"

        # Now log a query context for the same session
        mock_postgres_cursor.fetchone.return_value = {'id': 'context_456'}
        context_id = postgres_client.log_query_context(
            original_question="Test question?",
            session_id=session_id
        )

        # Verify context was logged and linked to the session
        assert context_id == "context_456"
        # Verify that the session_id was passed correctly to the INSERT statement
        args, kwargs = mock_postgres_cursor.execute.call_args
        assert session_id in str(args[0]) or session_id in str(args[1:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])