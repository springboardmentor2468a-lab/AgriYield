import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "N": 30,
    "P": 60,
    "K": 70,
    "temperature": 15,
    "humidity": 70,
    "ph": 7.5,
    "rainfall": 100,
    "requested_crop": "maize"
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
