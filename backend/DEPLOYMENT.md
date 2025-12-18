# Deployment Guide: Backend to Render

This guide walks you through deploying the Physical AI & Humanoid Robotics backend to Render.

---

## Prerequisites

Before deploying, ensure you have:

- ✅ GitHub repository with your code pushed
- ✅ [Render account](https://render.com) (free tier works)
- ✅ Neon Postgres database (already set up)
- ✅ Qdrant Cloud cluster (already set up)
- ✅ OpenAI API key

---

## Step 1: Prepare Environment Variables

Generate an AUTH_SECRET for production:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save this value - you'll need it in the Render dashboard.

**Required Environment Variables:**

| Variable | Example Value | Where to Get It |
|----------|---------------|-----------------|
| `DATABASE_URL` | `postgresql://user:pass@ep-...neon.tech/neondb?sslmode=require` | Neon Dashboard → Connection String (pooled) |
| `QDRANT_URL` | `https://90f39db9-...gcp.cloud.qdrant.io` | Qdrant Cloud → Cluster URL |
| `QDRANT_API_KEY` | `eyJhbGciOiJIUzI1NiI...` | Qdrant Cloud → API Keys |
| `OPENAI_API_KEY` | `sk-proj-...` | OpenAI Platform → API Keys |
| `AUTH_SECRET` | `rQvN7xK8mP2wYzJ5t...` | Generate with command above |
| `BETTER_AUTH_URL` | `https://your-app.vercel.app` | Your frontend URL (update after frontend deployment) |
| `CORS_ORIGINS` | `https://your-app.vercel.app,http://localhost:3000` | Your frontend URL (update after frontend deployment) |

---

## Step 2: Deploy to Render

### Option A: Deploy via Dashboard (Recommended)

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Click "New +" → "Web Service"

2. **Connect Your Repository**
   - Choose "Build and deploy from a Git repository"
   - Connect your GitHub account
   - Select your repository: `abdul-ahad-26/physical-ai-and-humanoid-robotics`

3. **Configure the Service**
   - **Name**: `physical-ai-backend` (or your preferred name)
   - **Region**: Oregon (or closest to your users)
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or Starter for production)

4. **Add Environment Variables**
   - Scroll down to "Environment Variables"
   - Click "Add Environment Variable" for each:
     ```
     DATABASE_URL = [your Neon connection string]
     QDRANT_URL = [your Qdrant URL]
     QDRANT_API_KEY = [your Qdrant API key]
     OPENAI_API_KEY = [your OpenAI API key]
     AUTH_SECRET = [generated secret]
     BETTER_AUTH_URL = https://your-app.vercel.app
     CORS_ORIGINS = https://your-app.vercel.app,http://localhost:3000
     PYTHON_VERSION = 3.11.0
     ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for initial build
   - Watch the logs for any errors

### Option B: Deploy via render.yaml

1. **Push Your Code**
   ```bash
   git add backend/render.yaml backend/requirements.txt
   git commit -m "feat: add Render deployment configuration"
   git push origin main
   ```

2. **Create Service from Blueprint**
   - Go to Render Dashboard
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Render will detect `render.yaml` automatically
   - Add environment variables in the dashboard
   - Click "Apply"

---

## Step 3: Initialize Database

After deployment, initialize your database tables:

1. **Get Your Backend URL**
   - In Render dashboard, copy your service URL
   - Example: `https://physical-ai-backend.onrender.com`

2. **Run Database Initialization**
   ```bash
   # Option 1: Use Render Shell (if on paid plan)
   # In Render dashboard → Shell tab
   python scripts/init_db.py

   # Option 2: Run locally with production DATABASE_URL
   cd backend
   DATABASE_URL="your-production-database-url" python scripts/init_db.py
   ```

---

## Step 4: Ingest Content

Populate Qdrant with textbook content:

```bash
cd backend

# Set production environment variables
export DATABASE_URL="your-production-database-url"
export QDRANT_URL="your-production-qdrant-url"
export QDRANT_API_KEY="your-qdrant-api-key"
export OPENAI_API_KEY="your-openai-api-key"

# Run ingestion
python scripts/direct_ingest.py
```

**Expected Output:**
```
🔄 Initializing Qdrant collection...
📚 Found 41 markdown files
🔄 Starting ingestion...
[1/41] Processing: capstone/code.md ... ✅ Created 48 chunks
...
✅ Ingestion complete!
   Files processed: 41
   Chunks created: 527
```

---

## Step 5: Test Your Deployment

### Test Health Endpoint
```bash
curl https://your-backend-url.onrender.com/health
```

Expected response:
```json
{"status": "healthy"}
```

### Test Authentication
```bash
# Sign up
curl -X POST https://your-backend-url.onrender.com/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "testpass123"}'
```

### Test Chat (after authentication)
```bash
curl -X POST https://your-backend-url.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -H "Cookie: session=YOUR_SESSION_COOKIE" \
  -d '{"message": "What is physical AI?", "session_id": null}'
```

---

## Step 6: Update Frontend Configuration

After backend is deployed, update your frontend:

1. **Update Environment Variable**
   ```bash
   # In your frontend .env or Vercel dashboard
   REACT_APP_API_URL=https://your-backend-url.onrender.com
   ```

2. **Update CORS in Render**
   - Go to Render dashboard → Environment
   - Update `CORS_ORIGINS` with your actual frontend URL
   - Update `BETTER_AUTH_URL` with your actual frontend URL
   - Click "Save Changes" (will redeploy)

---

## Troubleshooting

### Build Fails
- **Error: `requirements.txt not found`**
  - Check "Root Directory" is set to `backend`

- **Error: `Python version mismatch`**
  - Add `PYTHON_VERSION=3.11.0` to environment variables

### Runtime Errors
- **500 errors on `/health`**
  - Check logs: Render Dashboard → Logs tab
  - Usually missing environment variables

- **Database connection fails**
  - Verify `DATABASE_URL` is correct (use pooled connection)
  - Check Neon dashboard for IP allowlist (Render IPs should be allowed)

- **CORS errors in frontend**
  - Update `CORS_ORIGINS` with your actual frontend URL
  - Include both `https://` and no trailing slash

### Qdrant Connection Issues
- **`AsyncQdrantClient` errors**
  - Verify `QDRANT_URL` includes `https://`
  - Check `QDRANT_API_KEY` is correct
  - Test Qdrant connection in dashboard

### Slow First Request (Free Tier)
- **15-30 second delay on first request**
  - Expected on Render free tier (spins down after 15 min inactivity)
  - Upgrade to Starter plan ($7/month) for always-on service

---

## Production Checklist

Before going live:

- [ ] Database initialized with all tables
- [ ] Content ingested to Qdrant (527 chunks)
- [ ] All environment variables set correctly
- [ ] Frontend URL updated in `CORS_ORIGINS` and `BETTER_AUTH_URL`
- [ ] Health endpoint returns 200
- [ ] Can sign up new user
- [ ] Can log in
- [ ] Can send chat message and get response with citations
- [ ] Can log out
- [ ] HTTPS enabled (automatic on Render)
- [ ] Consider upgrading to Starter plan for production

---

## Monitoring

### View Logs
```bash
# In Render dashboard
Dashboard → Your Service → Logs tab

# Look for:
# - Startup messages
# - API requests
# - Errors or exceptions
```

### Check Metrics
```bash
# In Render dashboard
Dashboard → Your Service → Metrics tab

# Monitor:
# - CPU usage
# - Memory usage
# - Response times
# - Error rates
```

---

## Scaling

### Upgrade Plan
For production use, consider:
- **Starter Plan ($7/month)**: Always-on, no spin-down
- **Standard Plan ($25/month)**: More resources, better performance

### Optimize Performance
- Use Neon Postgres pooled connection
- Monitor Qdrant query performance
- Cache frequently asked questions (future enhancement)

---

## Support

- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Issues**: https://github.com/abdul-ahad-26/physical-ai-and-humanoid-robotics/issues

---

## Next Steps

After backend deployment:
1. Deploy frontend to Vercel
2. Update environment variables with production URLs
3. Test end-to-end flow
4. Monitor logs for first 24 hours
5. Set up alerts (optional)
