# HPHT Diamond Prediction API - PythonAnywhere Deployment Guide

## 🚀 Deploy to PythonAnywhere (Free)

### Prerequisites
- PythonAnywhere free account
- All project files ready

### Step 1: Create PythonAnywhere Account

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for a free account
3. Verify your email address
4. Login to your dashboard

### Step 2: Upload Files to PythonAnywhere

#### Method 1: Using Git (Recommended)

1. **Open a Bash console in PythonAnywhere:**
   - Go to "Consoles" tab
   - Click "Bash" to open a new console

2. **Clone your repository:**
   ```bash
   cd ~
   git clone https://github.com/jemin98/hpht_diamond_prediction.git
   cd hpht_diamond_prediction
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models:**
   ```bash
   python train_model.py
   ```

#### Method 2: Manual Upload

1. **Go to "Files" tab in PythonAnywhere**
2. **Create a new directory:**
   - Navigate to `/home/YOUR_USERNAME/`
   - Create folder: `hpht_diamond_prediction`
3. **Upload all files:**
   - Upload `app.py`
   - Upload `train_model.py`
   - Upload `requirements.txt`
   - Upload all other project files

### Step 3: Configure Web App

1. **Go to "Web" tab in PythonAnywhere**
2. **Click "Add a new web app"**
3. **Choose configuration:**
   - **Domain:** `YOUR_USERNAME.pythonanywhere.com`
   - **Framework:** `Manual configuration`
   - **Python version:** `3.9` (or latest available)

### Step 4: Configure WSGI File

1. **Click on the WSGI configuration file link**
2. **Replace the content with:**

```python
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
```

**Replace `YOUR_USERNAME` with your actual PythonAnywhere username**

### Step 5: Install Dependencies

1. **Go to "Consoles" tab**
2. **Open a Bash console**
3. **Navigate to your project:**
   ```bash
   cd ~/hpht_diamond_prediction
   ```

4. **Install requirements:**
   ```bash
   pip install fastapi uvicorn scikit-learn xgboost pandas numpy python-multipart pydantic requests joblib
   ```

5. **Train the models:**
   ```bash
   python train_model.py
   ```

### Step 6: Configure Virtual Environment (Optional but Recommended)

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python train_model.py
   ```

2. **Update WSGI file to use virtual environment:**
```python
import sys
import os

# Add virtual environment to path
venv_path = '/home/YOUR_USERNAME/hpht_diamond_prediction/venv/lib/python3.9/site-packages'
if venv_path not in sys.path:
    sys.path.append(venv_path)

# Add the project directory to Python path
path = '/home/YOUR_USERNAME/hpht_diamond_prediction'
if path not in sys.path:
    sys.path.append(path)

# Import the FastAPI app
from app import app

# For WSGI compatibility
application = app
```

### Step 7: Reload Web App

1. **Go back to "Web" tab**
2. **Click "Reload" button**
3. **Wait for the green "Live" status**

### Step 8: Test Your API

Your API will be available at: `https://YOUR_USERNAME.pythonanywhere.com`

Test it:

```bash
# Health check
curl https://YOUR_USERNAME.pythonanywhere.com/health

# Make a prediction
curl -X POST "https://YOUR_USERNAME.pythonanywhere.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "color_grade": "D",
    "clarity_grade": "FL",
    "cut_quality": "Excellent",
    "hue_level": "None",
    "carat_weight": 2.5
  }'
```

### Step 9: Update Your Flutter App

Update your Flutter Dio configuration:

```dart
class HPHTDiamondService {
  final Dio _dio = Dio();
  
  HPHTDiamondService() {
    // Replace with your PythonAnywhere URL
    _dio.options.baseUrl = 'https://YOUR_USERNAME.pythonanywhere.com';
    _dio.options.connectTimeout = const Duration(seconds: 30);
    _dio.options.receiveTimeout = const Duration(seconds: 30);
  }
}
```

## 📋 API Endpoints

- **Health Check:** `GET /health`
- **Prediction:** `POST /predict`
- **Statistics:** `GET /stats`
- **Documentation:** `GET /docs`

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors:**
   - Check that all dependencies are installed
   - Verify the path in WSGI file is correct

2. **Model Training Fails:**
   - Check PythonAnywhere console logs
   - Ensure you have enough disk space

3. **Web App Not Loading:**
   - Check the WSGI configuration
   - Reload the web app
   - Check error logs in "Web" tab

### Logs:
- View logs in "Web" tab → "Log files"
- Check error logs for debugging

## 💰 Cost

- **Free Tier:** Completely free
- **Paid Plans:** Start at $5/month for more resources

## 🔄 Updates

To update your deployed API:
1. Upload new files to PythonAnywhere
2. Retrain models if needed
3. Reload the web app

---

**Note:** PythonAnywhere free tier has some limitations but is perfect for testing and small applications. 