import requests

# Define the URL of the FastAPI endpoint
url = "http://localhost:8000/predict"

# Define the payload with iris measurements
payload = {
    "sepal_length": 10,
    "sepal_width": 15,
    "petal_length": 18,
    "petal_width": 20
}

# Send a POST request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Print the JSON response from the server
    print("Response:", response.json())
else:
    # Print the error if the request failed
    print("Failed to get a valid response. Status code:", response.status_code)
    print("Response content:", response.text)
