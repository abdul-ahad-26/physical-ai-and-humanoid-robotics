# ADR-003: Hybrid Rate Limiting Strategy

## Status
Accepted

## Date
2025-12-10

## Context
The API endpoints need protection against abuse while maintaining fair usage for legitimate users. Different types of abuse patterns may emerge, requiring a comprehensive approach to rate limiting that addresses various potential attack vectors and usage patterns.

## Decision
We will implement a hybrid rate limiting approach that combines multiple strategies:
- **Request-based limiting**: Limits the number of requests per time window
- **Token-based limiting**: Limits based on the number of tokens processed (addresses prompt injection attacks)
- **Concurrent request limiting**: Limits the number of simultaneous requests per user/IP
- **Sliding window approach**: Uses sliding windows rather than fixed windows for better protection

This multi-layered approach provides comprehensive protection against different types of potential abuse patterns while maintaining fair usage for legitimate users.

## Consequences
**Positive**:
- Comprehensive protection against various abuse patterns
- Flexible approach that can be tuned for different endpoints
- Prevents both simple request flooding and more sophisticated token-based attacks
- Sliding window provides better protection than fixed windows
- Can be configured differently for different endpoints (public vs protected)

**Negative**:
- More complex to implement and maintain than single method
- Requires more sophisticated tracking and state management
- Potential for false positives affecting legitimate users
- More complex configuration and monitoring

## Alternatives Considered
- **Simple request-based limiting**: Would provide less sophisticated protection
- **Token-based only**: Would not account for request concurrency issues
- **Fixed window vs sliding window**: Fixed windows can be bypassed more easily
- **API Key per request**: Would be too restrictive for legitimate usage
- **IP-based blocking only**: Would not address other types of abuse

## References
- specs/002-agentic-rag-backend/plan.md
- specs/002-agentic-rag-backend/research.md