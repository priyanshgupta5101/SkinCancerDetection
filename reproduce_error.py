import requests

url = 'http://127.0.0.1:5000/predict'
files = {'image': open('uploads/not_an_image.txt', 'rb')}

print("Sending POST request to /predict...")
try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    with open('response.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Response saved to response.html")
except Exception as e:
    print(f"Request failed: {e}")
