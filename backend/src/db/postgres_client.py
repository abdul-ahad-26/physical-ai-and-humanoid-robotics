import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

load_dotenv()

class PostgresClient:
    """
    Client for interacting with Neon Postgres database for session logging
    and analytics in the RAG + Agentic AI-Textbook Chatbot.
    """

    def __init__(self):
        self.connection_string = os.getenv("NEON_DB_URL")
        if not self.connection_string:
            raise ValueError("NEON_DB_URL environment variable is required")

        # Create connection pool with min 1 and max 20 connections
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=self.connection_string
            )
        except Exception as e:
            raise Exception(f"Failed to create connection pool: {str(e)}")

        # Initialize the database tables
        self._initialize_tables()

    def get_connection(self):
        """
        Get a database connection from the pool.
        """
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """
        Return a connection to the pool.
        """
        self.connection_pool.putconn(conn)

    def _initialize_tables(self):
        """
        Initialize the required database tables if they don't exist.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Create UserSession table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id VARCHAR(255),
                        query TEXT NOT NULL,
                        response TEXT,
                        retrieved_context JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_metadata JSONB DEFAULT '{}'
                    );
                """)

                # Create TextbookContent table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS textbook_content (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        content TEXT NOT NULL,
                        embeddings JSONB,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'deleted', 'archived'))
                    );
                """)

                # Create QueryContext table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_contexts (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        original_question TEXT NOT NULL,
                        highlight_override TEXT,
                        retrieved_chunks JSONB,
                        processed_context TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_id UUID REFERENCES user_sessions(id)
                    );
                """)

                # Create AgentExecutionLog table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_execution_logs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_id VARCHAR(255) NOT NULL,
                        tool_calls JSONB,
                        input_params JSONB,
                        output_result TEXT,
                        execution_time FLOAT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_id UUID REFERENCES user_sessions(id)
                    );
                """)

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_timestamp ON user_sessions(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_textbook_content_status ON textbook_content(status);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_contexts_session_id ON query_contexts(session_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_execution_logs_timestamp ON agent_execution_logs(timestamp);")

                conn.commit()
        finally:
            self.return_connection(conn)

    def log_user_session(self, query: str, response: str = None, user_id: str = None,
                        retrieved_context: List[Dict] = None, session_metadata: Dict = None) -> str:
        """
        Log a user session to the database.

        Args:
            query: The original user query
            response: The AI-generated response
            user_id: Optional user identifier
            retrieved_context: Context used to generate the response
            session_metadata: Additional metadata about the session

        Returns:
            The ID of the created session
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO user_sessions (user_id, query, response, retrieved_context, session_metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, query, response, json.dumps(retrieved_context) if retrieved_context else None,
                      json.dumps(session_metadata) if session_metadata else None))

                session_id = cursor.fetchone()['id']
                conn.commit()

                return str(session_id)
        finally:
            self.return_connection(conn)

    def get_user_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a user session by ID.

        Args:
            session_id: The ID of the session to retrieve

        Returns:
            The session data or None if not found
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM user_sessions WHERE id = %s", (session_id,))
                result = cursor.fetchone()

                return dict(result) if result else None
        finally:
            self.return_connection(conn)

    def log_query_context(self, original_question: str, session_id: str,
                         highlight_override: str = None, retrieved_chunks: List[Dict] = None,
                         processed_context: str = None) -> str:
        """
        Log query context to the database.

        Args:
            original_question: The original question from the user
            session_id: The associated session ID
            highlight_override: Optional highlighted text that replaced search context
            retrieved_chunks: List of content chunks retrieved from knowledge base
            processed_context: The final context sent to the AI model

        Returns:
            The ID of the created query context
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO query_contexts (original_question, highlight_override, retrieved_chunks,
                                              processed_context, session_id)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (original_question, highlight_override, json.dumps(retrieved_chunks) if retrieved_chunks else None,
                      processed_context, session_id))

                context_id = cursor.fetchone()['id']
                conn.commit()

                return str(context_id)
        finally:
            self.return_connection(conn)

    def log_agent_execution(self, agent_id: str, session_id: str, tool_calls: List[Dict] = None,
                           input_params: Dict = None, output_result: str = None,
                           execution_time: float = None) -> str:
        """
        Log agent execution to the database.

        Args:
            agent_id: The ID of the agent that executed
            session_id: The associated session ID
            tool_calls: List of tools called during execution
            input_params: Parameters passed to the agent
            output_result: Result returned by the agent
            execution_time: Time taken for execution in seconds

        Returns:
            The ID of the created execution log
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO agent_execution_logs (agent_id, session_id, tool_calls, input_params,
                                                    output_result, execution_time)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (agent_id, session_id, json.dumps(tool_calls) if tool_calls else None,
                      json.dumps(input_params) if input_params else None, output_result, execution_time))

                log_id = cursor.fetchone()['id']
                conn.commit()

                return str(log_id)
        finally:
            self.return_connection(conn)

    def cleanup_expired_sessions(self, days: int = 30) -> int:
        """
        Clean up expired sessions based on retention policy.

        Args:
            days: Number of days to retain sessions (default 30)

        Returns:
            Number of sessions deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM user_sessions
                    WHERE timestamp < %s
                """, (cutoff_date,))

                deleted_count = cursor.rowcount
                conn.commit()

                return deleted_count
        finally:
            self.return_connection(conn)

    def cleanup_old_query_contexts(self, cutoff_date: datetime) -> int:
        """
        Clean up old query contexts based on retention policy.

        Args:
            cutoff_date: DateTime before which contexts should be deleted

        Returns:
            Number of query contexts deleted
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM query_contexts
                    WHERE timestamp < %s
                """, (cutoff_date,))

                deleted_count = cursor.rowcount
                conn.commit()

                return deleted_count
        finally:
            self.return_connection(conn)

    def cleanup_old_agent_logs(self, cutoff_date: datetime) -> int:
        """
        Clean up old agent execution logs based on retention policy.

        Args:
            cutoff_date: DateTime before which logs should be deleted

        Returns:
            Number of agent logs deleted
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM agent_execution_logs
                    WHERE timestamp < %s
                """, (cutoff_date,))

                deleted_count = cursor.rowcount
                conn.commit()

                return deleted_count
        finally:
            self.return_connection(conn)

    def run_maintenance(self):
        """
        Run basic database maintenance operations.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Run ANALYZE to update table statistics
                cursor.execute("ANALYZE;")
                conn.commit()
        finally:
            self.return_connection(conn)

    def run_intensive_maintenance(self):
        """
        Run intensive database maintenance operations.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Run ANALYZE and VACUUM to optimize database
                cursor.execute("ANALYZE;")
                cursor.execute("VACUUM;")
                conn.commit()
        finally:
            self.return_connection(conn)

    def health_check(self) -> bool:
        """
        Check if the database is available.

        Returns:
            True if database is available, False otherwise
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception:
            return False
        finally:
            self.return_connection(conn)