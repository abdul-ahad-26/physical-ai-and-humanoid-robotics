# Phase 0 Research: User Personalization Feature

**Feature**: 005-user-personalization
**Date**: 2025-12-23
**Status**: Complete

---

## Research Summary

This document captures research findings for implementing user profiling, content personalization, and Urdu translation features.

---

## R1: OpenAI Agents SDK Implementation

### Decision
Use OpenAI Agents SDK (`openai-agents-python`) for both Personalization and Translation agents.

### Rationale
- Official OpenAI SDK with high-quality documentation (Benchmark Score: 90.9)
- Native async support matches FastAPI patterns
- Built-in Pydantic integration for structured outputs
- Guardrails support via `@input_guardrail` and `@output_guardrail`
- Simple agent composition with `Agent`, `Runner` primitives

### Key Patterns from Documentation

**Agent Creation with Structured Output:**
```python
from agents import Agent, Runner
from pydantic import BaseModel

class PersonalizedOutput(BaseModel):
    content: str
    adaptations_made: list[str]

agent = Agent(
    name="Personalization Agent",
    instructions="...",
    output_type=PersonalizedOutput,
    model="gpt-4o",
)

result = await Runner.run(agent, input_text)
output = result.final_output_as(PersonalizedOutput)
```

**Guardrail Pattern:**
```python
from agents import output_guardrail, GuardrailFunctionOutput

@output_guardrail
async def content_validation_guardrail(ctx, agent, output):
    # Validate output doesn't contain prohibited content
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"valid": True}
    )
```

### Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| OpenAI Agents SDK | Official, async, Pydantic, guardrails | Newer library | **SELECTED** |
| Raw OpenAI API | Full control, stable | Manual prompt management | Rejected |
| LangChain | Feature-rich | Heavy dependency, overkill | Rejected |

---

## R2: Session-Scoped Caching

### Decision
Use `cachetools.TTLCache` for in-memory session-scoped caching.

### Rationale
- Lightweight, zero external dependencies
- TTL-based expiration handles session cleanup
- Sufficient for <100 concurrent users
- Cache key pattern: `{user_id}:{chapter_id}:{content_type}`

### Implementation

```python
from cachetools import TTLCache

# Global caches with 1-hour TTL
personalization_cache = TTLCache(maxsize=1000, ttl=3600)
translation_cache = TTLCache(maxsize=1000, ttl=3600)

def get_cached_personalization(user_id: str, chapter_id: str) -> str | None:
    key = f"{user_id}:{chapter_id}:personalized"
    return personalization_cache.get(key)

def set_cached_personalization(user_id: str, chapter_id: str, content: str):
    key = f"{user_id}:{chapter_id}:personalized"
    personalization_cache[key] = content
```

### Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| cachetools.TTLCache | Simple, in-memory, TTL | Process-local | **SELECTED** |
| Redis | Distributed, persistent | External service, overkill | Rejected |
| Database | Persistent | Slow for frequent reads | Rejected |
| Browser localStorage | Client-side | Can't control TTL, security | Rejected |

---

## R3: RTL Support for Urdu

### Decision
CSS-based RTL with `dir="rtl"` attribute and isolated styling.

### Rationale
- Docusaurus supports RTL through standard CSS
- Only translated content container needs RTL, not entire page
- Code blocks must remain LTR even in RTL context
- Noto Nastaliq Urdu font provides proper script rendering

### Implementation

```css
/* frontend/src/css/rtl.css */
.translated-content-urdu {
  direction: rtl;
  text-align: right;
  font-family: 'Noto Nastaliq Urdu', 'Noto Sans Arabic', serif;
  line-height: 1.8;
}

/* Code blocks stay LTR */
.translated-content-urdu pre,
.translated-content-urdu code,
.translated-content-urdu .prism-code {
  direction: ltr;
  text-align: left;
}

/* Lists with RTL markers */
.translated-content-urdu ul,
.translated-content-urdu ol {
  padding-right: 2em;
  padding-left: 0;
}
```

### Font Loading

```html
<!-- Add to docusaurus.config.js stylesheets -->
<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">
```

---

## R4: Code Block Preservation

### Decision
Regex-based extraction before AI processing, placeholder replacement, restoration after.

### Rationale
- 100% guarantees code preservation
- AI never sees actual code, only placeholders
- Works for both fenced code blocks and inline code
- Simple, reliable pattern

### Implementation

```python
import re
from dataclasses import dataclass

@dataclass
class ExtractedContent:
    text: str
    code_blocks: list[str]
    inline_codes: list[str]

CODE_BLOCK_PATTERN = r'```[\s\S]*?```'
INLINE_CODE_PATTERN = r'`[^`\n]+`'

def extract_code(content: str) -> ExtractedContent:
    """Extract code and replace with placeholders."""
    # Extract fenced code blocks first
    code_blocks = re.findall(CODE_BLOCK_PATTERN, content)
    for i, block in enumerate(code_blocks):
        content = content.replace(block, f"[CODE_BLOCK_{i}]", 1)

    # Extract inline code
    inline_codes = re.findall(INLINE_CODE_PATTERN, content)
    for i, code in enumerate(inline_codes):
        content = content.replace(code, f"[INLINE_CODE_{i}]", 1)

    return ExtractedContent(
        text=content,
        code_blocks=code_blocks,
        inline_codes=inline_codes
    )

def restore_code(content: str, extracted: ExtractedContent) -> str:
    """Restore code from placeholders."""
    # Restore code blocks
    for i, block in enumerate(extracted.code_blocks):
        content = content.replace(f"[CODE_BLOCK_{i}]", block)

    # Restore inline code
    for i, code in enumerate(extracted.inline_codes):
        content = content.replace(f"[INLINE_CODE_{i}]", code)

    return content
```

### Validation

```python
def validate_code_preservation(original: str, processed: str) -> bool:
    """Verify all code blocks are preserved exactly."""
    original_blocks = re.findall(CODE_BLOCK_PATTERN, original)
    processed_blocks = re.findall(CODE_BLOCK_PATTERN, processed)
    return original_blocks == processed_blocks
```

---

## R5: Rate Limiting Strategy

### Decision
Per-user token bucket rate limiter for AI endpoints, 10 requests/minute.

### Rationale
- Controls AI API costs
- Prevents abuse
- Reuses existing middleware pattern from /api/chat
- Returns 429 with Retry-After header

### Implementation

```python
from collections import defaultdict
from time import time

class AIRateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """Check if request is allowed, return (allowed, retry_after_seconds)."""
        now = time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[user_id] = [
            t for t in self.requests[user_id] if t > window_start
        ]

        if len(self.requests[user_id]) >= self.max_requests:
            oldest = min(self.requests[user_id])
            retry_after = int(oldest + self.window_seconds - now) + 1
            return False, retry_after

        self.requests[user_id].append(now)
        return True, 0

ai_rate_limiter = AIRateLimiter(max_requests=10, window_seconds=60)
```

---

## R6: Profile Wizard UX

### Decision
Modal-based 3-step wizard with progress indicator and skip option.

### Rationale
- Modal doesn't interrupt navigation
- Progress indicator shows completion status
- Skip option preserves user agency
- Partial progress persisted on close

### State Management

```typescript
interface ProfileWizardState {
  step: 1 | 2 | 3;
  data: {
    displayName: string;
    softwareBackground: {
      level: 'beginner' | 'intermediate' | 'advanced';
      languages: string[];
      frameworks: string[];
    };
    hardwareBackground: {
      level: 'none' | 'basic' | 'intermediate' | 'advanced';
      domains: string[];
    };
  };
  isOpen: boolean;
  isSkipped: boolean;
}
```

### UI Components

1. **Step 1 - Display Name**: Text input with validation (required, max 100 chars)
2. **Step 2 - Software Background**:
   - Level dropdown (required)
   - Multi-select for languages
   - Multi-select for frameworks
3. **Step 3 - Hardware Background**:
   - Level dropdown (required)
   - Multi-select for domains

### Options Lists

```typescript
const SOFTWARE_LANGUAGES = [
  'Python', 'JavaScript', 'TypeScript', 'C++', 'Java',
  'Go', 'Rust', 'C', 'MATLAB', 'Julia'
];

const SOFTWARE_FRAMEWORKS = [
  'TensorFlow', 'PyTorch', 'React', 'FastAPI', 'ROS',
  'ROS2', 'Unity', 'Gazebo', 'Isaac Sim', 'OpenCV'
];

const HARDWARE_DOMAINS = [
  'basic_electronics', 'robotics_kits', 'gpus_accelerators',
  'jetson_edge', 'embedded_systems', 'sensors_actuators',
  '3d_printing', 'pcb_design'
];
```

---

## Research Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| openai-agents | latest | Agent implementation |
| cachetools | >=5.0 | TTL caching |
| pydantic | >=2.0 | Structured outputs |
| asyncpg | existing | Database operations |

---

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Which AI model for agents? | GPT-4o (balance of quality and speed) |
| Cache storage location? | In-memory (cachetools), not Redis |
| How to preserve code? | Pre-extraction with placeholders |
| RTL implementation? | CSS-based with dir attribute |
| Rate limit strategy? | 10 req/min per user |

---

## References

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python)
- [cachetools Documentation](https://cachetools.readthedocs.io/)
- [MDN RTL Guide](https://developer.mozilla.org/en-US/docs/Web/HTML/Global_attributes/dir)
- [Noto Nastaliq Urdu Font](https://fonts.google.com/noto/specimen/Noto+Nastaliq+Urdu)
