#!/usr/bin/env python3
"""
PythonAnywhere Deployment Script for HPHT Diamond Prediction API
"""

import os
import sys
import subprocess
import shutil

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_models():
    """Train the ML models"""
    print("🤖 Training ML models...")
    subprocess.run([sys.executable, "train_model.py"])

def create_wsgi_file():
    """Create WSGI file for PythonAnywhere"""
    wsgi_content = '''import sys
import os

# Add the project directory to Python path
path = '/home/YOUR_USERNAME/hpht_diamond_prediction'
if path not in sys.path:
    sys.path.append(path)

# Import the FastAPI app
from app import app

# For WSGI compatibility
application = app
'''
    
    with open('wsgi.py', 'w') as f:
        f.write(wsgi_content)
    print("📄 Created wsgi.py file")

def main():
    """Main deployment function"""
    print("🚀 Starting PythonAnywhere deployment...")
    
    # Install requirements
    install_requirements()
    
    # Train models
    train_models()
    
    # Create WSGI file
    create_wsgi_file()
    
    print("✅ Deployment completed!")
    print("\n📋 Next steps:")
    print("1. Upload all files to PythonAnywhere")
    print("2. Update wsgi.py with your username")
    print("3. Configure web app in PythonAnywhere dashboard")
    print("4. Set working directory to: /home/YOUR_USERNAME/hpht_diamond_prediction")
    print("5. Set WSGI configuration file to: /var/www/YOUR_USERNAME_pythonanywhere_com_wsgi.py")

if __name__ == "__main__":
    main() 