#!/usr/bin/env python3
"""
HPHT Diamond Treatment Prediction Model Training
Trains XGBoost model to predict ideal temperature and pressure for diamond treatment
Uses proper diamond color grades (D-Z), clarity grades (FL-I3), cut quality (Excellent-Poor), and hue levels (None-Dark)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import sqlite3
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Grade mappings for conversion to numeric values
COLOR_GRADE_MAPPING = {
    'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7, 'K': 8, 'L': 9, 'M': 10,
    'N': 11, 'O': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'U': 18, 'V': 19,
    'W': 20, 'X': 21, 'Y': 22, 'Z': 23
}

CLARITY_GRADE_MAPPING = {
    'FL': 1, 'IF': 2, 'VVS1': 3, 'VVS2': 4, 'VS1': 5, 'VS2': 6, 'SI1': 7, 'SI2': 8,
    'I1': 9, 'I2': 10, 'I3': 11
}

CUT_QUALITY_MAPPING = {
    'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5
}

HUE_LEVEL_MAPPING = {
    'None': 1, 'Faint': 2, 'Very Light': 3, 'Light': 4, 'Medium': 5, 'Dark': 6
}

def generate_synthetic_data(n_samples=50000):
    """
    Generate synthetic HPHT diamond treatment data with proper diamond grades
    """
    np.random.seed(42)
    
    # Generate proper diamond grades
    color_grades = list(COLOR_GRADE_MAPPING.keys())
    clarity_grades = list(CLARITY_GRADE_MAPPING.keys())
    cut_qualities = list(CUT_QUALITY_MAPPING.keys())
    hue_levels = list(HUE_LEVEL_MAPPING.keys())
    
    # Generate data with realistic distributions (probabilities sum to 1.0)
    # Color grades: D-Z (23 grades) - more common in middle grades
    color_probs = [0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    # Normalize to sum to 1.0
    color_probs = np.array(color_probs) / np.sum(color_probs)
    
    # Clarity grades: FL-I3 (11 grades) - more common in middle grades
    clarity_probs = [0.02, 0.03, 0.08, 0.12, 0.15, 0.20, 0.15, 0.12, 0.06, 0.04, 0.03]
    clarity_probs = np.array(clarity_probs) / np.sum(clarity_probs)
    
    # Cut qualities: Excellent-Poor (5 grades) - more common in middle grades
    cut_probs = [0.15, 0.25, 0.35, 0.20, 0.05]
    cut_probs = np.array(cut_probs) / np.sum(cut_probs)
    
    # Hue levels: None-Dark (6 levels) - more common in lighter hues
    hue_probs = [0.30, 0.25, 0.20, 0.15, 0.08, 0.02]
    hue_probs = np.array(hue_probs) / np.sum(hue_probs)
    
    data = {
        'color_grade': np.random.choice(color_grades, n_samples, p=color_probs),
        'clarity_grade': np.random.choice(clarity_grades, n_samples, p=clarity_probs),
        'cut_quality': np.random.choice(cut_qualities, n_samples, p=cut_probs),
        'hue_level': np.random.choice(hue_levels, n_samples, p=hue_probs),
        'carat_weight': np.random.uniform(0.2, 5.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Convert grades to numeric for model training
    df['color_grade_numeric'] = df['color_grade'].map(COLOR_GRADE_MAPPING)
    df['clarity_grade_numeric'] = df['clarity_grade'].map(CLARITY_GRADE_MAPPING)
    df['cut_quality_numeric'] = df['cut_quality'].map(CUT_QUALITY_MAPPING)
    df['hue_level_numeric'] = df['hue_level'].map(HUE_LEVEL_MAPPING)
    
    # Generate realistic temperature and pressure based on diamond characteristics
    # Higher grades generally require more precise treatment conditions
    base_temp = 1800
    base_pressure = 6.0
    
    # Temperature calculation based on grades
    temp_factor = (
        (df['color_grade_numeric'] - 1) * 5 +  # Color influence
        (df['clarity_grade_numeric'] - 1) * 3 +  # Clarity influence
        (6 - df['cut_quality_numeric']) * 4 +  # Cut influence (inverted)
        (7 - df['hue_level_numeric']) * 2 +  # Hue influence (inverted)
        (df['carat_weight'] - 2.6) * 20  # Weight influence
    )
    
    # Pressure calculation
    pressure_factor = (
        (df['color_grade_numeric'] - 1) * 0.02 +  # Color influence
        (df['clarity_grade_numeric'] - 1) * 0.015 +  # Clarity influence
        (6 - df['cut_quality_numeric']) * 0.025 +  # Cut influence (inverted)
        (7 - df['hue_level_numeric']) * 0.01 +  # Hue influence (inverted)
        (df['carat_weight'] - 2.6) * 0.05  # Weight influence
    )
    
    # Add some randomness for realistic variation
    temp_noise = np.random.normal(0, 30, n_samples)
    pressure_noise = np.random.normal(0, 0.1, n_samples)
    
    df['temperature'] = base_temp + temp_factor + temp_noise
    df['pressure'] = base_pressure + pressure_factor + pressure_noise
    
    # Clamp to realistic ranges
    df['temperature'] = np.clip(df['temperature'], 1500, 2200)
    df['pressure'] = np.clip(df['pressure'], 5.0, 7.0)
    
    return df

def create_database():
    """Create SQLite database with proper schema for string grades"""
    conn = sqlite3.connect('hpht_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            color_grade TEXT NOT NULL,
            clarity_grade TEXT NOT NULL,
            cut_quality TEXT NOT NULL,
            hue_level TEXT NOT NULL,
            carat_weight REAL NOT NULL,
            predicted_temperature REAL NOT NULL,
            predicted_pressure REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database schema created/updated")

def train_model(data):
    """
    Train XGBoost models for temperature and pressure prediction
    """
    # Prepare features (numeric values)
    X = data[['color_grade_numeric', 'clarity_grade_numeric', 'cut_quality_numeric', 'hue_level_numeric', 'carat_weight']].values
    y_temp = data['temperature'].values
    y_pressure = data['pressure'].values
    
    # Split data
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    _, _, y_pressure_train, y_pressure_test = train_test_split(X, y_pressure, test_size=0.2, random_state=42)
    
    # Train temperature model
    logger.info("Training temperature model...")
    temp_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    temp_model.fit(X_train, y_temp_train)
    
    # Train pressure model
    logger.info("Training pressure model...")
    pressure_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    pressure_model.fit(X_train, y_pressure_train)
    
    # Evaluate models
    temp_pred = temp_model.predict(X_test)
    pressure_pred = pressure_model.predict(X_test)
    
    temp_rmse = np.sqrt(mean_squared_error(y_temp_test, temp_pred))
    temp_r2 = r2_score(y_temp_test, temp_pred)
    temp_mae = mean_absolute_error(y_temp_test, temp_pred)
    
    pressure_rmse = np.sqrt(mean_squared_error(y_pressure_test, pressure_pred))
    pressure_r2 = r2_score(y_pressure_test, pressure_pred)
    pressure_mae = mean_absolute_error(y_pressure_test, pressure_pred)
    
    # Calculate percentage errors
    temp_rmse_pct = (temp_rmse / np.mean(y_temp_test)) * 100
    pressure_rmse_pct = (pressure_rmse / np.mean(y_pressure_test)) * 100
    
    metrics = {
        'temperature': {
            'rmse': temp_rmse,
            'rmse_pct': temp_rmse_pct,
            'r2': temp_r2,
            'mae': temp_mae
        },
        'pressure': {
            'rmse': pressure_rmse,
            'rmse_pct': pressure_rmse_pct,
            'r2': pressure_r2,
            'mae': pressure_mae
        }
    }
    
    return temp_model, pressure_model, metrics

def save_models_and_mappings(temp_model, pressure_model):
    """Save models and grade mappings"""
    # Save models
    joblib.dump(temp_model, 'temperature_model.pkl')
    joblib.dump(pressure_model, 'pressure_model.pkl')
    
    # Save grade mappings
    joblib.dump(COLOR_GRADE_MAPPING, 'color_grade_mapping.pkl')
    joblib.dump(CLARITY_GRADE_MAPPING, 'clarity_grade_mapping.pkl')
    joblib.dump(CUT_QUALITY_MAPPING, 'cut_quality_mapping.pkl')
    joblib.dump(HUE_LEVEL_MAPPING, 'hue_level_mapping.pkl')
    
    logger.info("Models and grade mappings saved successfully")

def main():
    """Main training function"""
    logger.info("Starting HPHT Diamond Treatment Prediction Model Training")
    
    # Generate synthetic data
    logger.info("Generating synthetic HPHT diamond data...")
    data = generate_synthetic_data(n_samples=50000)
    logger.info(f"Generated {len(data)} samples")
    
    # Create database
    create_database()
    
    # Train models
    temp_model, pressure_model, metrics = train_model(data)
    
    # Save models and mappings
    save_models_and_mappings(temp_model, pressure_model)
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Temperature Model - RMSE: {metrics['temperature']['rmse']:.1f}°C ({metrics['temperature']['rmse_pct']:.2f}%), R²: {metrics['temperature']['r2']:.4f}")
    logger.info(f"Pressure Model - RMSE: {metrics['pressure']['rmse']:.3f} GPa ({metrics['pressure']['rmse_pct']:.2f}%), R²: {metrics['pressure']['r2']:.4f}")
    
    # Check if RMSE is below 2%
    if metrics['temperature']['rmse_pct'] < 2.0 and metrics['pressure']['rmse_pct'] < 2.0:
        logger.info("✅ Both models achieve RMSE < 2% - Excellent accuracy!")
    elif metrics['temperature']['rmse_pct'] < 2.0:
        logger.info("✅ Temperature model achieves RMSE < 2%")
        logger.warning(f"⚠️ Pressure model RMSE: {metrics['pressure']['rmse_pct']:.2f}% (target: < 2%)")
    elif metrics['pressure']['rmse_pct'] < 2.0:
        logger.info("✅ Pressure model achieves RMSE < 2%")
        logger.warning(f"⚠️ Temperature model RMSE: {metrics['temperature']['rmse_pct']:.2f}% (target: < 2%)")
    else:
        logger.warning(f"⚠️ Temperature model RMSE: {metrics['temperature']['rmse_pct']:.2f}% (target: < 2%)")
        logger.warning(f"⚠️ Pressure model RMSE: {metrics['pressure']['rmse_pct']:.2f}% (target: < 2%)")

if __name__ == "__main__":
    main() 