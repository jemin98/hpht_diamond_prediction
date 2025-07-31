#!/bin/bash
echo "Starting HPHT Diamond Prediction API deployment..."

# Install dependencies
pip install -r requirements.txt

# Train the models
echo "Training ML models..."
python train_model.py

echo "Build completed successfully!" 