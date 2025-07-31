#!/usr/bin/env python3
"""
Test script for HPHT Diamond Treatment Prediction API
Demonstrates API usage with example requests
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def test_prediction(input_data: Dict[str, Any]):
    """Test the prediction endpoint"""
    print(f"\n🔮 Testing prediction with data: {input_data}")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print("✅ Prediction successful!")
            print(f"   Temperature: {prediction['predicted_temperature']}°C")
            print(f"   Pressure: {prediction['predicted_pressure']} GPa")
            print(f"   Confidence: {prediction['confidence_score']}")
            print(f"   Message: {prediction['message']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def test_stats():
    """Test the statistics endpoint"""
    print("\n📊 Testing statistics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Statistics retrieved successfully!")
            print(f"   Total predictions: {stats['total_predictions']}")
            print(f"   Average temperature: {stats['average_temperature']}°C")
            print(f"   Average pressure: {stats['average_pressure']} GPa")
            return True
        else:
            print(f"❌ Statistics failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def run_comprehensive_test():
    """Run a comprehensive test with multiple scenarios"""
    print("🚀 Starting HPHT Diamond Prediction API Test")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("\n❌ Health check failed. Please start the API server first.")
        print("Run: python app.py")
        return
    
    # Test various prediction scenarios
    test_cases = [
        {
            "name": "High Quality Diamond",
            "data": {
                "color_grade": 9,
                "clarity_grade": 9,
                "carat_weight": 3.0,
                "cut_quality": 5,
                "initial_hue_level": 30
            }
        },
        {
            "name": "Standard Quality Diamond",
            "data": {
                "color_grade": 6,
                "clarity_grade": 7,
                "carat_weight": 2.0,
                "cut_quality": 4,
                "initial_hue_level": 50
            }
        },
        {
            "name": "Low Quality Diamond",
            "data": {
                "color_grade": 3,
                "clarity_grade": 4,
                "carat_weight": 1.0,
                "cut_quality": 2,
                "initial_hue_level": 80
            }
        },
        {
            "name": "Large Diamond",
            "data": {
                "color_grade": 7,
                "clarity_grade": 8,
                "carat_weight": 4.5,
                "cut_quality": 4,
                "initial_hue_level": 40
            }
        },
        {
            "name": "Small Diamond",
            "data": {
                "color_grade": 8,
                "clarity_grade": 7,
                "carat_weight": 0.5,
                "cut_quality": 3,
                "initial_hue_level": 60
            }
        }
    ]
    
    successful_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}/{total_predictions}: {test_case['name']}")
        if test_prediction(test_case['data']):
            successful_predictions += 1
        time.sleep(0.5)  # Small delay between requests
    
    # Test statistics
    test_stats()
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 TEST SUMMARY")
    print(f"✅ Successful predictions: {successful_predictions}/{total_predictions}")
    print(f"📊 Success rate: {(successful_predictions/total_predictions)*100:.1f}%")
    
    if successful_predictions == total_predictions:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")

def test_invalid_inputs():
    """Test input validation"""
    print("\n🧪 Testing input validation...")
    
    invalid_cases = [
        {
            "name": "Invalid color grade (too high)",
            "data": {
                "color_grade": 15,
                "clarity_grade": 7,
                "carat_weight": 2.0,
                "cut_quality": 4,
                "initial_hue_level": 50
            }
        },
        {
            "name": "Invalid carat weight (too low)",
            "data": {
                "color_grade": 7,
                "clarity_grade": 7,
                "carat_weight": 0.1,
                "cut_quality": 4,
                "initial_hue_level": 50
            }
        },
        {
            "name": "Missing required field",
            "data": {
                "color_grade": 7,
                "clarity_grade": 7,
                "carat_weight": 2.0,
                "cut_quality": 4
                # Missing initial_hue_level
            }
        }
    ]
    
    for test_case in invalid_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 422:  # Validation error
                print("✅ Correctly rejected invalid input")
            else:
                print(f"⚠️  Unexpected response: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API")

if __name__ == "__main__":
    print("HPHT Diamond Treatment Prediction API Test Suite")
    print("=" * 60)
    
    # Run comprehensive test
    run_comprehensive_test()
    
    # Test input validation
    test_invalid_inputs()
    
    print("\n🏁 Test suite completed!")
    print("\n💡 To start the API server, run:")
    print("   python app.py")
    print("\n💡 To view API documentation, visit:")
    print("   http://localhost:8000/docs") 