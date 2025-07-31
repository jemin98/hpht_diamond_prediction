#!/bin/bash

echo "🚀 Deploying HPHT Diamond Prediction API to PythonAnywhere..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastapi uvicorn scikit-learn xgboost pandas numpy python-multipart pydantic requests joblib

# Train models
echo "🤖 Training ML models..."
python train_model.py

# Create WSGI file
echo "📄 Creating WSGI configuration..."
cat > wsgi.py << 'EOF'
import sys
import os

# Add the project directory to Python path
path = '/home/YOUR_USERNAME/hpht_diamond_prediction'
if path not in sys.path:
    sys.path.append(path)

# Import the FastAPI app
from app import app

# For WSGI compatibility
application = app
EOF

echo "✅ Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Update wsgi.py with your PythonAnywhere username"
echo "2. Go to Web tab in PythonAnywhere dashboard"
echo "3. Create new web app with Manual configuration"
echo "4. Set WSGI configuration file to point to your wsgi.py"
echo "5. Reload the web app"
echo ""
echo "🌐 Your API will be available at: https://YOUR_USERNAME.pythonanywhere.com" 