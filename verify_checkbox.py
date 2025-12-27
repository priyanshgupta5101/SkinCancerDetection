import requests

url = 'http://127.0.0.1:5000/predict'
files = {'image': open('uploads/debug_image.jpg', 'rb')}

# Test 1: Checkbox Checked (Expect Cancer)
print("Test 1: Checkbox Checked (Simulate Cancer)")
data = {'simulate_cancer': 'true'}
try:
    response = requests.post(url, files=files, data=data)
    if "Cancer" in response.text and "NonCancer" not in response.text:
        print("PASS: Result is Cancer")
    else:
        print("FAIL: Result is not Cancer")
        
    if "uploads/debug_image.jpg" in response.text:
         print("PASS: Image path present")
    else:
         print("FAIL: Image path missing")

except Exception as e:
    print(f"Test 1 Failed: {e}")

# Reset file pointer
files['image'].seek(0)

# Test 2: Checkbox Unchecked (Expect NonCancer)
print("\nTest 2: Checkbox Unchecked (Simulate NonCancer)")
data = {} # Checkbox not sent when unchecked
try:
    response = requests.post(url, files=files, data=data)
    if "NonCancer" in response.text:
        print("PASS: Result is NonCancer")
    else:
        print("FAIL: Result is not NonCancer")
except Exception as e:
    print(f"Test 2 Failed: {e}")

# Test 3: Image Access
print("\nTest 3: Image Access")
img_url = 'http://127.0.0.1:5000/uploads/debug_image.jpg'
try:
    response = requests.get(img_url)
    if response.status_code == 200:
        print("PASS: Image is accessible")
    else:
        print(f"FAIL: Image not accessible (Status: {response.status_code})")
except Exception as e:
    print(f"Test 3 Failed: {e}")
