import requests

url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
try:
    response = requests.head(url, allow_redirects=True, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Content-Length: {response.headers.get('Content-Length')}")
except Exception as e:
    print(f"Error: {e}")
