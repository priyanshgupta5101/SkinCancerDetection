import requests

url = 'http://127.0.0.1:5000/get'
params = {'msg': 'Hello'}

print("Sending GET request to /get...")
try:
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
