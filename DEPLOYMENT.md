# HPHT Diamond Prediction API - Render Deployment Guide

## 🚀 Deploy to Render

### Prerequisites
- Render account (free tier available)
- GitHub repository with your code

### Step 1: Prepare Your Repository

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: HPHT Diamond Prediction API"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/hpht-diamond-prediction.git
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. **Go to Render Dashboard:**
   - Visit [render.com](https://render.com)
   - Sign up/Login to your account

2. **Create New Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository: `hpht-diamond-prediction`

3. **Configure the Service:**
   - **Name:** `hpht-diamond-prediction-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables (Optional):**
   - `PYTHON_VERSION`: `3.13.0`

5. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- Render will automatically:
  - Install dependencies
  - Train the ML models
  - Start the server
  - Provide you with a URL (e.g., `https://your-app-name.onrender.com`)

### Step 4: Test Your API

Once deployed, test your API:

```bash
# Health check
curl https://your-app-name.onrender.com/health

# Make a prediction
curl -X POST "https://your-app-name.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "color_grade": "D",
    "clarity_grade": "FL",
    "cut_quality": "Excellent",
    "hue_level": "None",
    "carat_weight": 2.5
  }'
```

### Step 5: Update Your Flutter App

Update your Flutter Dio configuration to use the Render URL:

```dart
class HPHTDiamondService {
  final Dio _dio = Dio();
  
  HPHTDiamondService() {
    // Use your Render URL
    _dio.options.baseUrl = 'https://your-app-name.onrender.com';
    _dio.options.connectTimeout = const Duration(seconds: 30);
    _dio.options.receiveTimeout = const Duration(seconds: 30);
  }
  
  // ... rest of your service code
}
```

## 📋 API Endpoints

- **Health Check:** `GET /health`
- **Prediction:** `POST /predict`
- **Statistics:** `GET /stats`
- **Documentation:** `GET /docs`

## 🔧 Troubleshooting

### Common Issues:

1. **Build Fails:**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version is compatible

2. **Model Training Fails:**
   - The build script will retrain models during deployment
   - Check logs for any errors

3. **API Not Responding:**
   - Check if the service is running in Render dashboard
   - Verify the URL is correct

### Logs:
- View logs in Render dashboard
- Check for any error messages during build or runtime

## 💰 Cost

- **Free Tier:** 750 hours/month
- **Paid Plans:** Start at $7/month for always-on service

## 🔄 Updates

To update your deployed API:
1. Push changes to GitHub
2. Render will automatically redeploy
3. Models will be retrained during deployment

---

**Note:** The free tier may have cold starts (first request after inactivity takes longer). 