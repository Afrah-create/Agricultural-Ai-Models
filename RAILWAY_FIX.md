# Railway Deployment Fix

## The Problem
Railway is detecting your repo but looking at the root directory instead of the `deployment` folder.

## Solution: Set Root Directory in Railway

### Steps:
1. Go to https://railway.app
2. Open your project (Agri-Ai)
3. Go to **Settings** tab
4. Under **Service Settings**, find **Root Directory**
5. Change it from `/` to `/deployment`
6. Click **Save**
7. The deployment will automatically retry

### Alternative: Quick Fix
If the above doesn't work, you can also:

1. **Clone only the deployment folder**:
```bash
git clone https://github.com/Afrah-create/Agri-Ai.git
cd Agri-Ai/deployment
# Now upload just this folder to Railway
```

2. **Or create a new Railway project** pointing to:
   - GitHub repo: `Agri-Ai`
   - Branch: `main`
   - **Service Root**: `deployment`

## What Was Pushed:
- ✅ railway.toml - Tells Railway to use deployment folder
- ✅ nixpacks.toml - Build configuration
- ✅ deployment/railway.json - Deployment config
- ✅ Updated .gitignore

## Deployment Settings in Railway:
- **Root Directory**: `deployment`
- **Start Command**: `python app/main.py`
- **Environment Variables**: Add `PORT` (auto-set by Railway)

## If Still Having Issues:

### Option 1: Manual Deploy
1. Delete current Railway service
2. Create new service
3. Select "Deploy from GitHub repo"
4. Choose your repo
5. In **Settings → Service → Root Directory**, set to `deployment`

### Option 2: Railway CLI
```bash
railway init
# Choose your project
cd deployment
railway link
railway up
```

---
**Status**: Configuration files are in GitHub, ready for Railway! 🚀

