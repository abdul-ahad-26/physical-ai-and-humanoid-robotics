"""
Scheduled Cleanup Jobs for Session Data in RAG + Agentic Backend for AI-Textbook Chatbot.

This module implements scheduled cleanup jobs for session data including:
- Expired session cleanup
- Old query context cleanup
- Agent execution log cleanup
- Periodic data retention enforcement
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import logging
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import atexit

from .logging_agent import LoggingAgent
from .indexing_agent import IndexingAgent
from ..db.postgres_client import PostgresClient
from ..db.qdrant_client import QdrantClientWrapper


class CleanupScheduler:
    """
    Scheduler for automated cleanup jobs for session data and other temporary data.
    """

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.postgres_client = PostgresClient()
        self.qdrant_client = QdrantClientWrapper()
        self.logging_agent = LoggingAgent()
        self.indexing_agent = IndexingAgent()

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_scheduler(self):
        """Start the cleanup scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()

            # Schedule cleanup jobs
            self.schedule_cleanup_jobs()

            # Shut down the scheduler when exiting the app
            atexit.register(lambda: self.shutdown())

    def schedule_cleanup_jobs(self):
        """Schedule all cleanup jobs"""
        # Schedule daily cleanup at 2 AM
        self.scheduler.add_job(
            func=self.daily_cleanup,
            trigger=CronTrigger(hour=2, minute=0),  # Run daily at 2:00 AM
            id='daily_cleanup',
            name='Daily Cleanup Job',
            replace_existing=True
        )

        # Schedule hourly cleanup for temporary data
        self.scheduler.add_job(
            func=self.hourly_cleanup,
            trigger=IntervalTrigger(hours=1),
            id='hourly_cleanup',
            name='Hourly Cleanup Job',
            replace_existing=True
        )

        # Schedule weekly cleanup for logs
        self.scheduler.add_job(
            func=self.weekly_cleanup,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),  # Sunday at 3:00 AM
            id='weekly_cleanup',
            name='Weekly Cleanup Job',
            replace_existing=True
        )

        self.logger.info("Cleanup jobs scheduled successfully")

    def daily_cleanup(self):
        """Daily cleanup job"""
        self.logger.info("Starting daily cleanup job")

        try:
            # Clean up old user sessions (older than 30 days by default)
            old_sessions_deleted = self.clean_old_sessions(days=30)
            self.logger.info(f"Daily cleanup: Deleted {old_sessions_deleted} old user sessions")

            # Clean up old query contexts (older than 30 days)
            old_contexts_deleted = self.clean_old_query_contexts(days=30)
            self.logger.info(f"Daily cleanup: Deleted {old_contexts_deleted} old query contexts")

            # Clean up old agent execution logs (older than 30 days)
            old_logs_deleted = self.clean_old_agent_logs(days=30)
            self.logger.info(f"Daily cleanup: Deleted {old_logs_deleted} old agent execution logs")

            # Run database optimization
            self.optimize_database()
            self.logger.info("Daily cleanup: Database optimization completed")

        except Exception as e:
            self.logger.error(f"Error during daily cleanup: {e}")
            raise

        self.logger.info("Daily cleanup job completed")

    def hourly_cleanup(self):
        """Hourly cleanup job for temporary data"""
        self.logger.info("Starting hourly cleanup job")

        try:
            # Clean up temporary data that's older than 1 hour
            temp_data_cleaned = self.clean_temporary_data(hours=1)
            self.logger.info(f"Hourly cleanup: Cleaned {temp_data_cleaned} temporary items")

            # Clean up expired cache entries
            cache_entries_cleaned = self.clean_expired_cache()
            self.logger.info(f"Hourly cleanup: Cleaned {cache_entries_cleaned} expired cache entries")

        except Exception as e:
            self.logger.error(f"Error during hourly cleanup: {e}")

        self.logger.info("Hourly cleanup job completed")

    def weekly_cleanup(self):
        """Weekly cleanup job for logs and maintenance"""
        self.logger.info("Starting weekly cleanup job")

        try:
            # Clean up logs older than 90 days
            old_logs_deleted = self.clean_old_logs(days=90)
            self.logger.info(f"Weekly cleanup: Deleted {old_logs_deleted} old logs")

            # Clean up old indexing operations (older than 90 days)
            old_ops_deleted = self.clean_old_indexing_operations(days=90)
            self.logger.info(f"Weekly cleanup: Deleted {old_ops_deleted} old indexing operations")

            # Run more intensive database maintenance
            self.maintenance_database()
            self.logger.info("Weekly cleanup: Database maintenance completed")

        except Exception as e:
            self.logger.error(f"Error during weekly cleanup: {e}")

        self.logger.info("Weekly cleanup job completed")

    def clean_old_sessions(self, days: int = 30) -> int:
        """
        Clean up user sessions older than specified days.

        Args:
            days: Number of days to retain sessions

        Returns:
            Number of sessions deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        try:
            # Use the logging agent to perform cleanup
            result = self.logging_agent.cleanup_old_sessions(days=days)
            return result.get("deleted_count", 0) if isinstance(result, dict) else 0
        except Exception as e:
            self.logger.error(f"Error cleaning old sessions: {e}")
            return 0

    def clean_old_query_contexts(self, days: int = 30) -> int:
        """
        Clean up query contexts older than specified days.

        Args:
            days: Number of days to retain query contexts

        Returns:
            Number of query contexts deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        try:
            # Direct database cleanup for query contexts
            return self.postgres_client.cleanup_old_query_contexts(cutoff_date)
        except Exception as e:
            self.logger.error(f"Error cleaning old query contexts: {e}")
            return 0

    def clean_old_agent_logs(self, days: int = 30) -> int:
        """
        Clean up agent execution logs older than specified days.

        Args:
            days: Number of days to retain agent logs

        Returns:
            Number of agent logs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        try:
            # Direct database cleanup for agent logs
            return self.postgres_client.cleanup_old_agent_logs(cutoff_date)
        except Exception as e:
            self.logger.error(f"Error cleaning old agent logs: {e}")
            return 0

    def clean_temporary_data(self, hours: int = 1) -> int:
        """
        Clean up temporary data older than specified hours.

        Args:
            hours: Number of hours to retain temporary data

        Returns:
            Number of temporary items cleaned
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cleaned_count = 0

        try:
            # Clean up temporary files, cache entries, etc.
            # This is a placeholder implementation
            # In a real system, you'd clean up temporary files, cache, etc.
            self.logger.info(f"Cleaning temporary data older than {cutoff_time}")

            # For now, just return a placeholder
            cleaned_count = 0

        except Exception as e:
            self.logger.error(f"Error cleaning temporary data: {e}")

        return cleaned_count

    def clean_expired_cache(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of cache entries cleaned
        """
        cleaned_count = 0

        try:
            # In a real implementation, you'd connect to your caching system (Redis, etc.)
            # and remove expired entries
            # For now, this is a placeholder
            self.logger.info("Cleaning expired cache entries")

            # Placeholder implementation
            cleaned_count = 0

        except Exception as e:
            self.logger.error(f"Error cleaning expired cache: {e}")

        return cleaned_count

    def clean_old_logs(self, days: int = 90) -> int:
        """
        Clean up old system logs.

        Args:
            days: Number of days to retain logs

        Returns:
            Number of logs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0

        try:
            # Clean up old application logs
            # In a real implementation, you might clean up log files
            self.logger.info(f"Cleaning logs older than {cutoff_date}")

            # Placeholder implementation
            deleted_count = 0

        except Exception as e:
            self.logger.error(f"Error cleaning old logs: {e}")

        return deleted_count

    def clean_old_indexing_operations(self, days: int = 90) -> int:
        """
        Clean up old indexing operation records.

        Args:
            days: Number of days to retain indexing operations

        Returns:
            Number of indexing operations deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0

        try:
            # Clean up old indexing operation records from database
            # Placeholder implementation
            self.logger.info(f"Cleaning old indexing operations older than {cutoff_date}")

            # In a real implementation, you'd clean up indexing operation records
            deleted_count = 0

        except Exception as e:
            self.logger.error(f"Error cleaning old indexing operations: {e}")

        return deleted_count

    def optimize_database(self):
        """Run database optimization tasks"""
        try:
            # Run PostgreSQL maintenance operations
            self.postgres_client.run_maintenance()
            self.logger.info("Database optimization completed")
        except Exception as e:
            self.logger.error(f"Error during database optimization: {e}")

    def maintenance_database(self):
        """Run intensive database maintenance tasks"""
        try:
            # Run more intensive maintenance operations
            self.postgres_client.run_intensive_maintenance()
            self.logger.info("Database maintenance completed")
        except Exception as e:
            self.logger.error(f"Error during database maintenance: {e}")

    def shutdown(self):
        """Shutdown the scheduler gracefully"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("Cleanup scheduler shut down")


class CleanupJobManager:
    """
    Manager for cleanup jobs that can be used in the application lifecycle.
    """

    def __init__(self):
        self.cleanup_scheduler = CleanupScheduler()

    def initialize_cleanup_jobs(self):
        """Initialize and start cleanup jobs"""
        self.cleanup_scheduler.start_scheduler()
        print("Cleanup jobs initialized and started")

    def run_manual_cleanup(self, job_type: str = "all"):
        """
        Run manual cleanup for specific job type.

        Args:
            job_type: Type of cleanup to run ('daily', 'hourly', 'weekly', 'all')
        """
        if job_type == "daily" or job_type == "all":
            self.cleanup_scheduler.daily_cleanup()
        if job_type == "hourly" or job_type == "all":
            self.cleanup_scheduler.hourly_cleanup()
        if job_type == "weekly" or job_type == "all":
            self.cleanup_scheduler.weekly_cleanup()

    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about cleanup operations.

        Returns:
            Dictionary with cleanup statistics
        """
        # Placeholder implementation
        return {
            "last_cleanup_run": datetime.utcnow().isoformat(),
            "sessions_cleaned_today": 0,
            "contexts_cleaned_today": 0,
            "logs_cleaned_today": 0,
            "next_scheduled_cleanup": "2025-12-12T02:00:00Z"
        }


# Global cleanup manager instance
cleanup_manager = CleanupJobManager()


def initialize_cleanup_jobs():
    """Initialize cleanup jobs - call this during application startup"""
    cleanup_manager.initialize_cleanup_jobs()


def run_cleanup_job(job_type: str = "all"):
    """
    Run a manual cleanup job.

    Args:
        job_type: Type of cleanup job to run ('daily', 'hourly', 'weekly', 'all')
    """
    cleanup_manager.run_manual_cleanup(job_type)


def get_cleanup_stats() -> Dict[str, Any]:
    """Get cleanup statistics"""
    return cleanup_manager.get_cleanup_statistics()


# Example usage
if __name__ == "__main__":
    print("Initializing cleanup scheduler...")

    # Initialize cleanup jobs
    initialize_cleanup_jobs()

    # Example of running manual cleanup
    print("\nRunning manual daily cleanup...")
    run_cleanup_job("daily")

    # Get statistics
    stats = get_cleanup_stats()
    print(f"\nCleanup statistics: {stats}")

    print("\nCleanup scheduler is running. Press Ctrl+C to stop.")

    try:
        # Keep the scheduler running
        while True:
            time.sleep(60)  # Sleep for 1 minute
    except KeyboardInterrupt:
        print("\nShutting down cleanup scheduler...")
        cleanup_manager.cleanup_scheduler.shutdown()
        print("Cleanup scheduler stopped.")