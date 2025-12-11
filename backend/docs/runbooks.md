# Runbooks: RAG + Agentic Backend for AI-Textbook Chatbot

## Overview

This document provides runbooks for common operational tasks for the RAG + Agentic Backend for AI-Textbook Chatbot. Each runbook includes step-by-step procedures, escalation paths, and recovery actions.

## Table of Contents

1. [System Startup](#system-startup)
2. [System Shutdown](#system-shutdown)
3. [Health Checks and Monitoring](#health-checks-and-monitoring)
4. [Database Operations](#database-operations)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)
6. [Security Incidents](#security-incidents)
7. [Performance Tuning](#performance-tuning)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scaling Operations](#scaling-operations)
10. [Release Management](#release-management)

---

## System Startup

### Purpose
Start the RAG + Agentic Backend service and all required dependencies.

### Prerequisites
- All environment variables are configured (see `.env.example`)
- Dependencies (PostgreSQL, Qdrant, Redis) are running
- Proper resource allocation (CPU, Memory, Disk)

### Steps

1. **Verify Dependencies**
   ```bash
   # Check if PostgreSQL is running
   pg_isready -h $NEON_DB_URL

   # Check if Qdrant is running
   curl -X GET $QDRANT_URL/health

   # Check if Redis is running (if used)
   redis-cli ping
   ```

2. **Set Environment Variables**
   ```bash
   export $(grep -v '^#' .env | xargs)
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Database Migrations** (if needed)
   ```bash
   # If using database migrations
   python -m alembic upgrade head
   ```

5. **Start the Application**
   ```bash
   # For development
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

   # For production
   gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

6. **Verify Service Health**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/api/v1/health
   ```

### Expected Results
- Application starts without errors
- Health endpoints return "healthy" status
- All dependencies are accessible
- Logs show successful startup

### Troubleshooting
- If dependencies are not running, start them first
- Check environment variables are properly set
- Review application logs for specific errors

---

## System Shutdown

### Purpose
Safely stop the RAG + Agentic Backend service and handle any cleanup.

### Steps

1. **Graceful Shutdown**
   ```bash
   # If running with uvicorn
   # Send SIGTERM signal to allow graceful shutdown
   pkill -TERM uvicorn

   # If running with gunicorn
   pkill -TERM gunicorn
   ```

2. **Wait for Cleanup**
   - Allow 30 seconds for ongoing requests to complete
   - Check application logs for completion messages

3. **Force Shutdown (if needed)**
   ```bash
   # If graceful shutdown fails
   pkill -9 uvicorn
   pkill -9 gunicorn
   ```

4. **Verify Shutdown**
   ```bash
   # Check that process is no longer running
   ps aux | grep uvicorn
   ps aux | grep gunicorn
   ```

### Expected Results
- All ongoing requests are completed or gracefully terminated
- Resources are properly released
- No orphaned processes remain

---

## Health Checks and Monitoring

### Purpose
Monitor system health and performance metrics.

### Health Check Endpoints

1. **Application Health**
   ```bash
   curl -X GET http://localhost:8000/health
   curl -X GET http://localhost:8000/api/v1/health
   ```

2. **Component Health**
   ```bash
   # Query endpoint health
   curl -X GET http://localhost:8000/api/v1/query/health

   # Answer endpoint health
   curl -X GET http://localhost:8000/api/v1/answer/health

   # Index endpoint health
   curl -X GET http://localhost:8000/api/v1/index/health

   # Ready endpoint
   curl -X GET http://localhost:8000/api/v1/health/ready
   ```

### Monitoring Commands

1. **Resource Utilization**
   ```bash
   # Monitor application resources
   top -p $(pgrep -f uvicorn)

   # Check disk usage
   df -h

   # Check memory usage
   free -h
   ```

2. **Log Monitoring**
   ```bash
   # Tail application logs
   tail -f logs/app.log

   # Search for errors
   grep -i error logs/app.log

   # Search for warnings
   grep -i warning logs/app.log
   ```

3. **Performance Metrics**
   ```bash
   # Check response times
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/query

   # Or use a monitoring tool
   ab -n 100 -c 10 http://localhost:8000/api/v1/query
   ```

### Key Metrics to Watch

- **Response Times**: Should be <2s for /answer, <300ms for /query
- **Error Rates**: Should be <1%
- **CPU Usage**: Should be <80%
- **Memory Usage**: Should be stable
- **Database Connections**: Should not exceed pool size
- **Qdrant Availability**: Should remain accessible

---

## Database Operations

### Purpose
Perform common database maintenance and operational tasks.

### Backup Database

1. **Create Backup**
   ```bash
   # For PostgreSQL/Neon
   pg_dump $NEON_DB_URL > backup_$(date +%Y%m%d_%H%M%S).sql
   ```

2. **Verify Backup**
   ```bash
   # Check backup file size and contents
   ls -lh backup_*.sql
   head -20 backup_*.sql
   ```

### Restore Database

1. **Stop Application**
   - Follow shutdown procedure above

2. **Restore from Backup**
   ```bash
   # For PostgreSQL/Neon
   psql $NEON_DB_URL < backup_file.sql
   ```

3. **Verify Restoration**
   ```bash
   # Check tables exist
   psql $NEON_DB_URL -c "\dt"

   # Check row counts
   psql $NEON_DB_URL -c "SELECT COUNT(*) FROM user_sessions;"
   ```

### Index Maintenance

1. **Check Qdrant Collections**
   ```bash
   curl -X GET "$QDRANT_URL/collections"
   ```

2. **Optimize Collections**
   ```bash
   # Force segment optimization (be cautious in production)
   curl -X POST "$QDRANT_URL/collections/textbook_content/points/scroll" \
     -H "Content-Type: application/json" \
     -d '{"limit": 1}'
   ```

### Cleanup Operations

1. **Run Session Cleanup**
   ```bash
   # This would be handled by the scheduled cleanup job
   # Check the LoggingAgent for cleanup functionality
   python -c "
   from src.agent.logging_agent import LoggingAgent
   agent = LoggingAgent()
   result = agent.cleanup_old_sessions(days=30)
   print(f'Cleaned up {result} old sessions')
   "
   ```

---

## Troubleshooting Common Issues

### High Response Times

**Symptoms**:
- Response times >2s for /answer endpoint
- Response times >300ms for /query endpoint

**Steps**:
1. Check Qdrant availability:
   ```bash
   curl -X GET $QDRANT_URL/health
   ```
2. Check database connections:
   ```bash
   psql $NEON_DB_URL -c "SELECT count(*) FROM pg_stat_activity;"
   ```
3. Check system resources:
   ```bash
   top
   free -h
   ```
4. Review application logs for slow queries or operations

### Service Unavailable (503)

**Symptoms**:
- 503 errors from API endpoints
- Qdrant or database unavailable

**Steps**:
1. Check Qdrant status:
   ```bash
   curl -X GET $QDRANT_URL/collections
   ```
2. Check database connectivity:
   ```bash
   pg_isready -h $NEON_DB_URL
   ```
3. Check system logs:
   ```bash
   journalctl -u rag-backend --since "1 hour ago"
   ```
4. Restart failed dependencies if needed

### Authentication Failures

**Symptoms**:
- 401 Unauthorized errors
- Invalid API key messages

**Steps**:
1. Verify API key is correctly set:
   ```bash
   echo $INDEXING_API_KEY
   ```
2. Check authentication middleware:
   ```bash
   curl -H "Authorization: Bearer $INDEXING_API_KEY" \
        -X GET http://localhost:8000/api/v1/index/health
   ```
3. Review auth middleware logs

### Rate Limiting Issues

**Symptoms**:
- 429 Too Many Requests errors
- Requests being blocked unexpectedly

**Steps**:
1. Check current rate limiting configuration:
   ```bash
   # Check environment variables
   echo $RATE_LIMIT_REQUESTS_PER_MINUTE
   echo $RATE_LIMIT_TOKENS_PER_MINUTE
   echo $RATE_LIMIT_CONCURRENT_REQUESTS
   ```
2. Review rate limiting logs
3. Adjust limits if needed

---

## Security Incidents

### Purpose
Respond to security incidents and vulnerabilities.

### Suspicious Activity Detection

**Symptoms**:
- Multiple failed authentication attempts
- Unusual traffic patterns
- Suspicious input patterns

**Steps**:
1. **Immediate Response**:
   - Check security logs for suspicious patterns
   - Review recent authentication failures
   - Identify source IP addresses of suspicious activity

2. **Investigation**:
   ```bash
   # Check for brute force attempts
   grep "AUTHENTICATION_FAILURE" logs/app.log | tail -20

   # Check for suspicious input
   grep "INPUT_VALIDATION_FAILURE" logs/app.log | tail -20
   ```

3. **Mitigation**:
   - Block malicious IP addresses if confirmed
   - Increase monitoring on affected endpoints
   - Review and strengthen security controls

### Data Breach Response

**Steps**:
1. **Containment**:
   - Isolate affected systems
   - Disable compromised accounts/API keys
   - Preserve evidence

2. **Assessment**:
   - Determine scope of breach
   - Identify data accessed/exposed
   - Assess potential impact

3. **Recovery**:
   - Rotate all affected credentials
   - Patch security vulnerabilities
   - Restore from clean backup if needed

4. **Communication**:
   - Notify appropriate stakeholders
   - Document incident for post-mortem

---

## Performance Tuning

### Purpose
Optimize system performance for better response times and throughput.

### Caching Optimization

1. **Check Cache Hit Ratios**:
   ```bash
   # If using Redis for caching
   redis-cli info | grep -E "(keyspace|hit)"
   ```

2. **Adjust Cache TTLs**:
   - For frequently asked questions: 1-2 hours
   - For query results: 15-30 minutes
   - For embeddings: 30 minutes - 1 hour

### Database Connection Pooling

1. **Monitor Connection Usage**:
   ```bash
   psql $NEON_DB_URL -c "SELECT state, count(*) FROM pg_stat_activity GROUP BY state;"
   ```

2. **Adjust Pool Size**:
   - Modify `pool_size` and `max_overflow` in database client
   - Monitor for connection timeouts

### Qdrant Performance

1. **Check Collection Performance**:
   ```bash
   curl -X GET "$QDRANT_URL/collections/textbook_content/points/scroll?limit=1"
   ```

2. **Optimize Segment Configuration**:
   - Adjust indexing parameters based on data size
   - Consider HNSW parameters for faster search

---

## Backup and Recovery

### Purpose
Maintain reliable backup and recovery procedures.

### Daily Backup Procedure

1. **Automated Backup**:
   ```bash
   # Add to crontab for daily backups
   0 2 * * * pg_dump $NEON_DB_URL > /backups/daily_$(date +\%Y\%m\%d).sql
   ```

2. **Verify Backup Integrity**:
   ```bash
   # Check backup file
   head -10 /backups/daily_*.sql
   ```

### Recovery Testing

1. **Test Recovery Monthly**:
   - Restore backup to test environment
   - Verify data integrity
   - Document recovery time

2. **Update Recovery Procedures**:
   - Review and update runbooks based on tests
   - Train team on recovery procedures

---

## Scaling Operations

### Purpose
Scale the system to handle increased load.

### Horizontal Scaling

1. **Add Application Instances**:
   ```bash
   # Start additional workers
   gunicorn src.api.main:app -w 8 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
   ```

2. **Load Balancer Configuration**:
   - Add new instances to load balancer
   - Configure health checks
   - Monitor distribution

### Vertical Scaling

1. **Increase Resources**:
   - Monitor current resource usage
   - Plan resource increases based on trends
   - Schedule maintenance window if needed

2. **Qdrant Scaling**:
   - Consider Qdrant cluster setup for high availability
   - Adjust segment and indexing parameters

---

## Release Management

### Purpose
Manage application releases and deployments safely.

### Pre-Deployment Checklist

- [ ] All tests pass
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Rollback plan prepared
- [ ] Stakeholders notified

### Deployment Steps

1. **Prepare Deployment**:
   ```bash
   # Create deployment branch/tag
   git checkout -b deploy-$(date +%Y%m%d)

   # Build Docker image
   docker build -t rag-backend:$(date +%Y%m%d_%H%M%S) .
   ```

2. **Deploy to Staging**:
   - Deploy to staging environment
   - Run smoke tests
   - Verify functionality

3. **Deploy to Production**:
   - Deploy using blue-green or canary approach
   - Monitor health and metrics
   - Rollback if issues arise

### Rollback Procedure

1. **Identify Issues**:
   - Monitor for error spikes
   - Check response times
   - Review user reports

2. **Initiate Rollback**:
   ```bash
   # If using Docker
   docker service rollback rag-backend

   # If using Kubernetes
   kubectl rollout undo deployment/rag-backend
   ```

3. **Verify Rollback**:
   - Confirm service stability
   - Monitor metrics
   - Document lessons learned

---

## Emergency Contacts

- **On-Call Engineer**: [Contact Information]
- **Security Team**: [Contact Information]
- **Infrastructure Team**: [Contact Information]
- **Management Escalation**: [Contact Information]

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-12-11 | 1.0 | Initial runbook creation | AI-Textbook Team |
| [Date] | [Version] | [Changes] | [Author] |

---

**Important Notes**:
- This runbook should be reviewed and updated regularly
- Procedures should be tested in non-production environments
- Team members should be trained on these procedures
- Update this document as systems evolve