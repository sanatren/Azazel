# 🚂 Deploy Azazel AI to Railway

Complete guide to deploy Azazel AI on Railway with both backend and frontend.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Frontend       │ ───────▶│  Backend API    │
│  (Static HTML)  │         │  (FastAPI)      │
│  Port: 8080     │         │  Port: 8000     │
└─────────────────┘         └─────────────────┘
```

## Prerequisites

1. Railway account (https://railway.app)
2. GitHub repository with Azazel AI code
3. API Keys:
   - OpenAI API Key (users will provide their own)
   - Google API Key (for web search)
   - Google CSE ID (for web search)
   - Supabase URL & Key (optional, for chat history)

## Step 1: Deploy Backend (FastAPI)

### 1.1 Create New Project

1. Go to Railway dashboard
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your Azazel AI repository

### 1.2 Configure Backend Service

1. **Service Name**: `azazel-api`

2. **Root Directory**: `/api`

3. **Build Command**: (Auto-detected from Dockerfile)

4. **Start Command**:
   ```
   uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```

5. **Environment Variables**:
   ```
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_google_cse_id
   SUPABASE_URL=your_supabase_url (optional)
   SUPABASE_KEY=your_supabase_key (optional)
   ```

6. **Port**: Railway will auto-assign

### 1.3 Deploy

1. Click **"Deploy"**
2. Wait for deployment to complete
3. Copy the generated URL (e.g., `https://azazel-api.up.railway.app`)

## Step 2: Deploy Frontend

### 2.1 Create New Service

1. In the same Railway project
2. Click **"New"** → **"Service"** → **"GitHub Repo"**
3. Select the same repository

### 2.2 Configure Frontend Service

1. **Service Name**: `azazel-frontend`

2. **Root Directory**: `/frontend`

3. **Start Command**:
   ```
   python server.py
   ```

4. **Environment Variables**:
   ```
   PORT=8080
   ```

5. **Generate Domain**: Click "Generate Domain" to get a public URL

### 2.3 Update API URL

1. Go to your frontend service settings
2. Add environment variable:
   ```
   API_URL=https://your-backend-url.up.railway.app
   ```

OR

1. Edit `frontend/static/js/config.js`:
   ```javascript
   const CONFIG = {
       API_URL: 'https://your-backend-url.up.railway.app',
       // ...
   };
   ```

2. Commit and push changes

### 2.4 Deploy

1. Railway will auto-deploy on commit
2. Visit your frontend URL
3. You're live! 🎉

## Step 3: Configure Custom Domain (Optional)

### 3.1 For Frontend

1. Go to frontend service settings
2. Click **"Settings"** → **"Domains"**
3. Click **"Custom Domain"**
4. Enter your domain (e.g., `azazel.yourdomain.com`)
5. Add CNAME record to your DNS:
   ```
   CNAME  azazel  your-railway-url.up.railway.app
   ```

### 3.2 For Backend

1. Same process for backend service
2. Use subdomain like `api.yourdomain.com`
3. Update frontend `API_URL` to use custom domain

## Step 4: Environment Setup

### Required Environment Variables

**Backend (`azazel-api`)**:
```env
GOOGLE_API_KEY=AIzaSy...           # For web search
GOOGLE_CSE_ID=0123456789...        # For web search
SUPABASE_URL=https://xxx.supabase.co  # Optional
SUPABASE_KEY=eyJhbGci...           # Optional
```

**Frontend (`azazel-frontend`)**:
```env
PORT=8080
```

## Step 5: Testing

1. **Visit Frontend URL**
2. **Enter OpenAI API Key** in sidebar
3. **Send a test message**
4. **Try features**:
   - Regular chat
   - Web search (toggle in sidebar)
   - Code execution
   - Document upload

## Monitoring

### Backend Logs

1. Go to `azazel-api` service
2. Click **"Deployments"**
3. Click latest deployment
4. View real-time logs

### Frontend Logs

Same process for `azazel-frontend` service

## Scaling

Railway auto-scales based on usage. For heavy load:

1. Go to service settings
2. Adjust **"Resources"**:
   - Memory: 512MB - 8GB
   - CPU: Shared - Dedicated

## Cost Estimation

- **Free Tier**: $5 credit/month
- **Hobby Plan**: $5/month
- **Pro Plan**: $20/month

Typical usage:
- Backend: ~$2-5/month
- Frontend: ~$1-2/month
- **Total**: ~$3-7/month

## Troubleshooting

### Backend Not Starting

1. Check environment variables are set
2. View deployment logs
3. Verify Dockerfile is correct
4. Check Railway dashboard for errors

### Frontend Can't Connect to Backend

1. Verify `API_URL` in config.js
2. Check CORS settings in backend
3. Ensure both services are running
4. Test backend health: `https://your-api-url/health`

### 503 Service Unavailable

1. Service may be sleeping (free tier)
2. First request wakes it up (~30s)
3. Upgrade to Hobby plan for always-on

## Alternative: One-Click Deploy

### Backend

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### Frontend

After backend is deployed, create a new service and deploy frontend.

## CI/CD

Railway automatically:
- Builds on every push to main branch
- Deploys on successful build
- Rolls back on failure

To disable auto-deploy:
1. Go to service settings
2. Toggle off **"Auto Deploy"**

## Backup & Recovery

### Database Backup (if using Supabase)

Supabase provides automatic backups

### Code Backup

Always keep your GitHub repo as source of truth

## Security Best Practices

1. **API Keys**: Never commit API keys to git
2. **Environment Variables**: Use Railway's encrypted env vars
3. **HTTPS**: Railway provides free SSL
4. **CORS**: Configure allowed origins in production
5. **Rate Limiting**: Implement in FastAPI for production

## Monitoring & Analytics

### Add Analytics (Optional)

1. **Google Analytics**:
   ```html
   <!-- Add to frontend/index.html -->
   <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
   ```

2. **Sentry** (Error Tracking):
   ```bash
   pip install sentry-sdk
   ```

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Azazel AI Issues: Your GitHub repo issues

## Success Checklist

- [ ] Backend deployed and accessible
- [ ] Frontend deployed and accessible
- [ ] Environment variables configured
- [ ] API URL updated in frontend
- [ ] Test message sent successfully
- [ ] Web search working
- [ ] Document upload working
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up

## Next Steps

1. Share your app URL 🚀
2. Collect user feedback
3. Monitor usage and performance
4. Scale as needed
5. Add custom features

---

**Congratulations! Your Azazel AI is now live on Railway! 🎉**