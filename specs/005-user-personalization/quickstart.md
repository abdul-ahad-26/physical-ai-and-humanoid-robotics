# Quickstart Guide: User Personalization Feature

**Feature**: 005-user-personalization
**Date**: 2025-12-23

This guide helps developers get started with implementing and testing the user personalization feature.

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL database (Neon Serverless)
- OpenAI API key with GPT-4o access

---

## 1. Environment Setup

### Backend

Add the following to `backend/.env`:

```bash
# Existing variables (should already be set)
DATABASE_URL=postgresql://...
OPENAI_API_KEY=sk-...

# New variables for this feature
OPENAI_MODEL=gpt-4o
PERSONALIZATION_CACHE_TTL=3600
TRANSLATION_CACHE_TTL=3600
```

### Frontend

No new environment variables required for frontend.

---

## 2. Install Dependencies

### Backend

```bash
cd backend

# If using uv
uv pip install openai-agents cachetools

# If using pip
pip install openai-agents cachetools
```

### Frontend

```bash
cd frontend

# No new dependencies required - uses existing React setup
npm install  # If dependencies need refresh
```

---

## 3. Run Database Migration

```bash
cd backend

# Option 1: Run migration script
python -c "
import asyncio
from src.db.connection import get_db_pool

async def migrate():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(open('migrations/005_add_profile_fields.sql').read())
    print('Migration complete!')

asyncio.run(migrate())
"

# Option 2: Direct psql (if you have psql installed)
psql \$DATABASE_URL -f migrations/005_add_profile_fields.sql
```

---

## 4. Verify Migration

```bash
# Check tables exist
psql $DATABASE_URL -c "
SELECT
    'users.software_background' as field,
    EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='users' AND column_name='software_background') as exists
UNION ALL SELECT 'personalization_logs table', EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='personalization_logs')
UNION ALL SELECT 'translation_logs table', EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='translation_logs');
"
```

Expected output:
```
         field          | exists
------------------------+--------
 users.software_background | t
 personalization_logs table | t
 translation_logs table     | t
```

---

## 5. Start Development Servers

### Terminal 1: Backend

```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

### Terminal 2: Frontend

```bash
cd frontend
npm start
```

---

## 6. Test the Feature

### 6.1 Create a Test User

1. Go to http://localhost:3000/signup
2. Create account with email/password or OAuth

### 6.2 Test Profile API

```bash
# Get profile (should show profile_completed: false)
curl -X GET http://localhost:8000/api/profile \
  -H "Cookie: session=YOUR_SESSION_TOKEN"

# Update profile
curl -X POST http://localhost:8000/api/profile \
  -H "Content-Type: application/json" \
  -H "Cookie: session=YOUR_SESSION_TOKEN" \
  -d '{
    "display_name": "Test User",
    "software_background": {
      "level": "intermediate",
      "languages": ["Python", "JavaScript"],
      "frameworks": ["TensorFlow"]
    },
    "hardware_background": {
      "level": "basic",
      "domains": ["basic_electronics"]
    }
  }'
```

### 6.3 Test Personalization API

```bash
curl -X POST http://localhost:8000/api/personalize \
  -H "Content-Type: application/json" \
  -H "Cookie: session=YOUR_SESSION_TOKEN" \
  -d '{
    "chapter_id": "test-chapter",
    "content": "# Neural Networks\n\nNeural networks are computational models inspired by biological neurons.",
    "title": "Test Chapter"
  }'
```

### 6.4 Test Translation API

```bash
curl -X POST http://localhost:8000/api/translate \
  -H "Content-Type: application/json" \
  -H "Cookie: session=YOUR_SESSION_TOKEN" \
  -d '{
    "chapter_id": "test-chapter",
    "content": "# Neural Networks\n\nNeural networks are computational models.",
    "target_language": "ur"
  }'
```

---

## 7. Development Tips

### Testing Agents Locally

Create a test script `backend/test_agents.py`:

```python
import asyncio
from src.agents.personalization import personalize_content, UserProfile
from src.agents.translation import translate_content

async def test_personalization():
    profile = UserProfile(
        software_level="beginner",
        software_languages=["Python"],
        software_frameworks=[],
        hardware_level="none",
        hardware_domains=[]
    )

    result = await personalize_content(
        content="# Test\n\nThis is a test about neural networks.",
        profile=profile,
        chapter_title="Test Chapter"
    )
    print("Personalized:", result.content[:200])
    print("Adaptations:", result.adaptations_made)

async def test_translation():
    result = await translate_content(
        content="# Test\n\nThis is a test about neural networks.\n\n```python\nprint('hello')\n```"
    )
    print("Translated:", result.content[:200])
    print("Preserved blocks:", result.preserved_blocks)

if __name__ == "__main__":
    asyncio.run(test_personalization())
    asyncio.run(test_translation())
```

Run with:
```bash
cd backend
python test_agents.py
```

### Clearing Cache (Development)

```python
# In Python REPL or script
from src.services.cache import personalization_cache, translation_cache

personalization_cache.clear()
translation_cache.clear()
```

### Checking Rate Limits

```python
from src.api.middleware import ai_rate_limiter

# Check if user can make request
allowed, retry_after = ai_rate_limiter.is_allowed("user-123")
print(f"Allowed: {allowed}, Retry after: {retry_after}s")
```

---

## 8. Common Issues

### Issue: "Profile must be completed before personalizing content"

**Cause**: User's `profile_completed` is false.

**Solution**: Update profile with both `software_background.level` and `hardware_background.level` set.

### Issue: "Rate limit exceeded"

**Cause**: More than 10 requests in 60 seconds.

**Solution**: Wait for rate limit window to reset, or adjust `AIRateLimiter` settings for development.

### Issue: AI timeout errors

**Cause**: OpenAI API taking longer than 30 seconds.

**Solution**:
1. Check OpenAI status page
2. Try smaller content chunks
3. Increase timeout in development

### Issue: Code blocks modified after personalization

**Cause**: Code extraction not working properly.

**Solution**: Check that content uses standard markdown code fences (```).

---

## 9. File Structure Reference

```
backend/
├── src/
│   ├── api/
│   │   ├── profile.py         # Profile endpoints
│   │   ├── personalize.py     # Personalization endpoint
│   │   └── translate.py       # Translation endpoint
│   ├── agents/
│   │   ├── personalization.py # Personalization agent
│   │   └── translation.py     # Translation agent
│   ├── services/
│   │   └── cache.py           # TTLCache service
│   └── db/
│       ├── models.py          # Pydantic models
│       └── queries.py         # Database queries
├── migrations/
│   └── 005_add_profile_fields.sql
└── tests/
    └── unit/
        ├── test_profile.py
        └── test_agents.py

frontend/
├── src/
│   ├── components/
│   │   ├── ProfileWizard/
│   │   └── ChapterActions/
│   ├── pages/
│   │   └── profile.tsx
│   └── lib/
│       └── personalization.ts
└── tests/
```

---

## 10. Next Steps

1. **Run /sp.tasks** to generate detailed implementation tasks
2. Start with database migration
3. Implement backend endpoints in order: profile → personalize → translate
4. Implement frontend components in order: ProfileWizard → ChapterActions → ProfilePage
5. Run integration tests
6. Deploy and monitor

---

## Support

- Check `specs/005-user-personalization/plan.md` for architecture details
- Check `specs/005-user-personalization/research.md` for technology decisions
- Check `specs/005-user-personalization/contracts/` for API specifications
