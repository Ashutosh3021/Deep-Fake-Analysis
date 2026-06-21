# Deploy DeepGuard AI to Render

This guide will walk you through deploying the DeepGuard AI deepfake detection system to Render, a cloud platform for hosting web applications and APIs.

---

## Prerequisites

Before beginning, make sure you have:
1. A Render account (sign up at [render.com](https://render.com))
2. Your project hosted on a Git repository (GitHub, GitLab, or Bitbucket)
3. A valid Gemini API key (obtainable from [Google AI Studio](https://aistudio.google.com/app/apikey))

---

## Step 1: Prepare Your Repository

First, ensure your repository has all required files:
- `render.yaml` (already created for infrastructure as code)
- `Procfile` (alternative for deployment configuration)
- `requirements.txt` (with all dependencies)
- `.python-version` (specifies Python 3.10.12 for Render)
- `backend/api.py` (Flask app)
- `models/` directory (with detection models)
- `dashboard/` directory (with frontend files)
- `.env.example` (for environment variable reference)

---

## Step 2: Create a New Render Service

There are two ways to deploy on Render: using `render.yaml` (infrastructure as code) or manually through the dashboard. We'll cover both methods.

### Method A: Using render.yaml (Recommended)

1. Push your code to your Git repository.
2. Log into your Render account.
3. Click on "Blueprints" in the Render dashboard navigation.
4. Click "New Blueprint Instance".
5. Connect your Git repository and select the branch you want to deploy.
6. Review the blueprint configuration and click "Apply".
7. Wait for the service to build and deploy (this may take several minutes for the first deployment).

### Method B: Manual Deployment

1. Log into your Render account.
2. Click "New" → "Web Service" from the dashboard.
3. Connect your Git repository and select the appropriate branch.
4. Configure your web service with these settings:
   - **Name**: `deepguard-ai` (or your preferred name)
   - **Runtime**: Python 3
   - **Region**: Choose one closest to your users
   - **Branch**: The branch you want to deploy (e.g., main/master)
   - **Root Directory**: Leave empty or set to your repo root
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd backend && gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT api:app`
5. Click "Create Web Service".

---

## Step 3: Configure Environment Variables

After creating your service, you need to set up environment variables:

1. Go to your service dashboard in Render.
2. Click on "Environment" in the left sidebar.
3. Click "Add Environment Variable".
4. Add the following environment variable:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: Your actual Gemini API key
5. Save your changes.

---

## Step 4: Deploy and Verify

Once your service is created and environment variables are set, Render will automatically start building and deploying your application.

### Post-Deployment Verification Steps

1. Check the Render service logs to ensure the app started successfully.
2. Navigate to your service URL (provided by Render, e.g., `https://your-service-name.onrender.com`).
3. Test each functionality:
   - Image upload and deepfake detection
   - Video upload and deepfake detection
   - Audio upload and deepfake detection
   - Text analysis
   - Query Assistant (image and text queries)

---

## Troubleshooting Common Issues

### Issue 1: Build Failures

**Possible causes**:
- Missing dependencies
- Incompatible Python versions
- Network issues downloading dependencies

**Solutions**:
- Check Render logs for specific error messages
- Verify `requirements.txt` includes all dependencies
- Ensure the Python version in your `render.yaml` matches what your code needs

### Issue 2: Environment Variable Errors

**Possible causes**:
- Missing GEMINI_API_KEY
- Incorrectly set environment variable

**Solutions**:
- Double-check that GEMINI_API_KEY is set in Render's environment variables
- Verify the API key is valid

### Issue 3: High Memory Usage

**Possible causes**:
- Large machine learning models loading
- Multiple concurrent requests

**Solutions**:
- Consider upgrading your Render plan
- Ensure models are loaded lazily (already implemented in this project)

### Issue 4: Slow Initial Response

**Possible causes**:
- Machine learning models downloading on first run
- Cold start

**Solutions**:
- The first request will be slower as models are downloaded to Render's cache; subsequent requests will be faster
- Consider using a larger instance for better performance

---

## Additional Tips

1. **Custom Domain**: You can add a custom domain to your Render service from the "Custom Domains" section in the dashboard.
2. **Automatic Deployments**: Enable automatic deployments so your service updates whenever you push to your Git repository.
3. **Scaling**: For higher traffic, you can scale your service horizontally or vertically from the Render dashboard.
4. **Monitoring**: Use Render's built-in logs and metrics to monitor your application's performance.

---

## Cleanup

To delete your service (if you no longer need it):
1. Go to your service dashboard in Render.
2. Click "Settings" in the left sidebar.
3. Scroll down to "Delete Web Service" and confirm the deletion.
