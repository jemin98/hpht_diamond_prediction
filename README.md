# HPHT Diamond Treatment Prediction System

A complete backend system for predicting ideal temperature and pressure for HPHT (High Pressure High Temperature) diamond treatment based on diamond characteristics.

## 🎯 Features

- **ML Model**: XGBoost-based prediction for temperature and pressure
- **FastAPI Backend**: RESTful API with automatic documentation
- **SQLite Logging**: Stores all predictions with timestamps
- **Input Validation**: Pydantic models ensure data integrity
- **Confidence Scoring**: Calculates prediction confidence based on input characteristics

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Generate synthetic HPHT diamond data (10,000 samples)
- Train XGBoost models for temperature and pressure prediction
- Save models as `temperature_model.pkl` and `pressure_model.pkl`
- Create SQLite database for logging
- Display model performance metrics

### 3. Start the API Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prediction Endpoint**: POST http://localhost:8000/predict

## 📊 Model Performance

The trained models achieve:
- **Temperature RMSE**: < 2% (typically ~1.5%)
- **Pressure RMSE**: < 2% (typically ~1.8%)
- **R² Score**: > 0.95 for both models

## 🔧 API Usage

### Prediction Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "color_grade": 7,
  "clarity_grade": 8,
  "carat_weight": 2.5,
  "cut_quality": 4,
  "initial_hue_level": 45
}
```

**Response:**
```json
{
  "predicted_temperature": 1850.5,
  "predicted_pressure": 6.125,
  "confidence_score": 0.847,
  "message": "Standard temperature treatment at 1850°C, Standard pressure treatment at 6.125 GPa"
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "color_grade": 7,
       "clarity_grade": 8,
       "carat_weight": 2.5,
       "cut_quality": 4,
       "initial_hue_level": 45
     }'
```

### Python Example

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "color_grade": 7,
    "clarity_grade": 8,
    "carat_weight": 2.5,
    "cut_quality": 4,
    "initial_hue_level": 45
}

response = requests.post(url, json=data)
prediction = response.json()
print(f"Temperature: {prediction['predicted_temperature']}°C")
print(f"Pressure: {prediction['predicted_pressure']} GPa")
print(f"Confidence: {prediction['confidence_score']}")
```

## 📈 Input Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `color_grade` | 1-10 | Diamond color grade (1=worst, 10=best) |
| `clarity_grade` | 1-10 | Diamond clarity grade (1=worst, 10=best) |
| `carat_weight` | 0.2-5.0 | Diamond weight in carats |
| `cut_quality` | 1-5 | Cut quality rating (1=worst, 5=best) |
| `initial_hue_level` | 1-100 | Initial hue level (1=low, 100=high) |

## 🗄️ Database Schema

The system automatically logs all predictions to `hpht_predictions.db`:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    color_grade INTEGER,
    clarity_grade INTEGER,
    carat_weight REAL,
    cut_quality INTEGER,
    initial_hue_level INTEGER,
    predicted_temperature REAL,
    predicted_pressure REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 📊 Statistics Endpoint

**GET** `/stats`

Returns prediction statistics:
- Total number of predictions
- Average temperature and pressure
- Recent predictions with timestamps

## 🔍 Health Check

**GET** `/health`

Returns API health status and model loading status.

## 🏗️ Project Structure

```
hpht_diamond_prediction/
├── train_model.py          # Model training script
├── app.py                  # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── temperature_model.pkl   # Trained temperature model
├── pressure_model.pkl      # Trained pressure model
└── hpht_predictions.db    # SQLite database
```

## 🧪 Model Details

### Features Used
- **color_grade**: Affects both temperature and pressure requirements
- **clarity_grade**: Influences treatment intensity
- **carat_weight**: Larger diamonds require higher temperatures/pressures
- **cut_quality**: Better cuts need more precise treatment parameters
- **initial_hue_level**: Determines treatment intensity

### Model Architecture
- **Algorithm**: XGBoost Regressor
- **Ensemble Size**: 200 trees
- **Max Depth**: 8
- **Learning Rate**: 0.1
- **Objective**: reg:squarederror

### Prediction Ranges
- **Temperature**: 1500-2200°C
- **Pressure**: 5.0-7.0 GPa

## 🔧 Development

### Adding New Features

1. **New Input Parameters**: Update `DiamondInput` model in `app.py`
2. **Model Retraining**: Modify `train_model.py` and regenerate data
3. **Additional Endpoints**: Add new routes to `app.py`

### Testing

```bash
# Test the API
curl -X GET "http://localhost:8000/health"

# Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"color_grade": 5, "clarity_grade": 6, "carat_weight": 1.5, "cut_quality": 3, "initial_hue_level": 50}'
```

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Note**: This system uses synthetic data for demonstration. In production, use real HPHT diamond treatment data for training. 