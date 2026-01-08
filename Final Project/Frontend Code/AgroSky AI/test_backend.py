"""
Simple test script to verify backend is working
Run this after starting the Flask server
"""

import requests
import json

# Test data
test_data = {
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "temperature": 20.9,
    "humidity": 82.0,
    "rainfall": 202.9,
    "ph": 6.5
}

def test_backend():
    base_url = "http://127.0.0.1:5000"
    
    print("=" * 50)
    print("Testing AgroSky AI Backend")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend!")
        print("   Make sure you ran: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Prediction endpoint
    print("\n2. Testing Prediction Endpoint...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Recommended Crop: {result.get('recommended_crop')}")
            print(f"   Predicted Yield: {result.get('predicted_yield_kg_per_ha')} kg/ha")
            print(f"   Model: {result.get('model')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend!")
        print("   Make sure you ran: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\nMake sure Flask server is running (python app.py)\n")
    input("Press Enter to start testing...")
    test_backend()
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

