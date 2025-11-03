# 🚀 Quick Deploy Guide

Deploy Azazel AI to Railway in minutes!

## Option 1: Railway (Recommended) - Easiest

### Backend Deployment

1. **Create Railway Project**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your Azazel repository

2. **Configure Backend Service**
   - Railway will auto-detect the Dockerfile
   - Set these environment variables:
     ```
     GOOGLE_API_KEY=your_google_key
     GOOGLE_CSE_ID=your_cse_id
     SUPABASE_URL=your_supabase_url (optional)
     SUPABASE_KEY=your_supabase_key (optional)
     ```

3. **Deploy**
   - Railway auto-builds and deploys
   - Copy the generated URL (e.g., `https://azazel.railway.app`)

### Frontend Deployment

1. **Add Frontend Service**
   - In same Railway project: "New" → "GitHub Repo"
   - Select same repository

2. **Configure Settings**
   - **Root Directory**: `frontend`
   - **Start Command**: `python server.py`
   - **Port**: Railway auto-assigns

3. **Update API URL**
   - Go to frontend service
   - Add environment variable:
     ```
     API_URL=https://your-backend-url.railway.app
     ```
   - Redeploy

4. **Generate Domain**
   - Click "Settings" → "Generate Domain"
   - Your app is live! 🎉

## Option 2: Docker Compose (Local/VPS)

```bash
# Clone repo
git clone your-repo-url
cd Azazel

# Create .env file
cat > .env << EOF
GOOGLE_API_KEY=your_key
GOOGLE_CSE_ID=your_cse_id
EOF

# Start everything
docker-compose up -d

# Access
# Frontend: http://localhost:8080
# Backend: http://localhost:8000
```

## Option 3: Render

### Backend
1. Create Web Service
2. **Build Command**: `pip install -r api/requirements.txt`
3. **Start Command**: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables

### Frontend
1. Create Static Site
2. **Build Command**: (none)
3. **Publish Directory**: `frontend`
4. Update `config.js` with backend URL

## Environment Variables

### Required
- `GOOGLE_API_KEY` - For web search
- `GOOGLE_CSE_ID` - For web search

### Optional
- `SUPABASE_URL` - For chat history
- `SUPABASE_KEY` - For chat history

## Testing

1. Visit frontend URL
2. Enter OpenAI API key in sidebar
3. Send test message: "Hello!"
4. Try features:
   - ✅ Regular chat
   - ✅ Web search (toggle in sidebar)
   - ✅ Code execution
   - ✅ Document upload

## Troubleshooting

### Backend fails to start
```bash
# Check logs on Railway
# Verify environment variables are set
# Ensure Dockerfile is in project root
```

### Frontend can't connect
```bash
# Update API_URL in frontend/static/js/config.js
# Check CORS settings
# Verify backend is running
```

### Build fails
```bash
# Make sure Dockerfile context is project root
# Verify all Bot modules exist
# Check requirements.txt has all dependencies
```

## Cost

Railway:
- Free tier: $5 credit/month
- Hobby: $5/month
- Estimated cost: $3-7/month for both services

## Support

- [Railway Docs](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)
- Create issue on GitHub

## Quick Links

- Backend Health: `https://your-api-url/health`
- API Docs: `https://your-api-url/docs`
- Frontend: `https://your-frontend-url`

---

**That's it! Your AI assistant is live! 🚀**