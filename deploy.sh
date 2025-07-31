#!/bin/bash

echo "🚀 Preparing HPHT Diamond Prediction API for Render deployment..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy HPHT Diamond Prediction API to Render"

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Please add your GitHub repository as remote origin:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/hpht-diamond-prediction.git"
    echo ""
    echo "Then run: git push -u origin main"
else
    echo "🚀 Pushing to GitHub..."
    git push origin main
fi

echo ""
echo "✅ Repository ready for Render deployment!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://render.com"
echo "2. Create new Web Service"
echo "3. Connect your GitHub repository"
echo "4. Use these settings:"
echo "   - Build Command: pip install -r requirements.txt && python train_model.py"
echo "   - Start Command: uvicorn app:app --host 0.0.0.0 --port \$PORT"
echo ""
echo "🌐 Your API will be available at: https://your-app-name.onrender.com" 