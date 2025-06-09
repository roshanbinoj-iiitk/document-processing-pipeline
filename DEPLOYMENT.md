# 🚀 Deployment Guide - Document Processing Pipeline

## GitHub Repository Setup

### Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Document Processing Pipeline with Streamlit"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in repository details:
   - **Repository name**: `document-processing-pipeline`
   - **Description**: `AI-powered document processing with OCR and vision models`
   - **Visibility**: Public (or Private if preferred)
   - **DO NOT** initialize with README, .gitignore, or license (we already have them)
5. Click "Create repository"

### Step 3: Connect Local Repository to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/document-processing-pipeline.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Deployment Options

### Option 1: Streamlit Cloud (Recommended) 🌟

#### Prerequisites:
- GitHub repository (created above)
- Streamlit Cloud account

#### Steps:

1. **Visit Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy New App**
   - Click "New app"
   - Select your GitHub repository: `document-processing-pipeline`
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Configure Secrets**
   - In your deployed app, go to "Settings" → "Secrets"
   - Add your Groq API key:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

4. **System Dependencies**
   - The `packages.txt` file will automatically install:
     - tesseract-ocr
     - tesseract-ocr-eng
     - poppler-utils

#### ✅ **Streamlit Cloud Features:**
- Free hosting for public repositories
- Automatic deployments on git push
- Built-in secrets management
- Custom domain support
- SSL certificates included

### Option 2: Heroku

#### Additional Files Needed:

1. **Create Procfile**
```bash
echo "web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0" > Procfile
```

2. **Create Aptfile** (for system dependencies)
```bash
echo -e "tesseract-ocr\ntesseract-ocr-eng\npoppler-utils" > Aptfile
```

3. **Deploy to Heroku**
```bash
# Install Heroku CLI first, then:
heroku create your-app-name
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks:add --index 2 heroku/python
git push heroku main
```

### Option 3: Docker + Cloud Run/DigitalOcean

#### Create Dockerfile:
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Environment Variables Setup

### For Streamlit Cloud:
- Use the built-in secrets management
- Add `GROQ_API_KEY` in the secrets section

### For Other Platforms:
- Set environment variable: `GROQ_API_KEY=your_api_key`
- Update app.py to use `os.getenv('GROQ_API_KEY')` as fallback

## Update App for Production

Modify the API key input in `app.py` to use environment variables:

```python
# In the sidebar section, update:
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv('GROQ_API_KEY', ''),  # Use env var as default
    type="password",
    help="Enter your Groq API key to use the vision and text models"
)
```

## File Structure for Deployment

```
document-processing-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies (for Streamlit Cloud)
├── README.md             # Project documentation
├── DEPLOYMENT.md         # This deployment guide
├── run.sh               # Local development script
├── .gitignore           # Git ignore file
└── pyproject.toml       # uv project file
```

## Testing Deployment

1. **Local Testing**
   ```bash
   # Test with production requirements
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **Check Dependencies**
   - Ensure all imports work
   - Test OCR functionality
   - Verify API connections

## Monitoring and Maintenance

- **Streamlit Cloud**: Monitor through the dashboard
- **Logs**: Check application logs for errors
- **Updates**: Push changes to trigger automatic redeployment
- **Secrets**: Rotate API keys regularly

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Keep dependencies updated** regularly
4. **Monitor usage** to prevent abuse
5. **Set appropriate file upload limits**

## Cost Considerations

- **Streamlit Cloud**: Free for public repos
- **Heroku**: Free tier available (with limitations)
- **Cloud providers**: Pay-per-use (typically $5-20/month)

---

## Quick Start Commands

```bash
# 1. Initialize and push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/document-processing-pipeline.git
git push -u origin main

# 2. Deploy to Streamlit Cloud
# Visit share.streamlit.io and connect your repository

# 3. Add secrets in Streamlit Cloud dashboard
# GROQ_API_KEY = "your_key_here"
```

🎉 **Your app will be live and accessible worldwide!**

