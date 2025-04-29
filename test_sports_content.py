import requests
import json
import time

# URL for the sentiment analysis endpoint
url = "http://127.0.0.1:5000/analyze"

# Test sports-related content that was causing errors
sports_test_data = [
    {
        "text": "Manchester United's decision to hire Amorim seems like a smart move for the club's future.",
        "description": "Sports content about Manchester United and Amorim"
    },
    {
        "text": "Amorim might struggle at Manchester United given the club's recent managerial challenges.",
        "description": "Critical sports content about Manchester United and Amorim"
    },
    {
        "text": "Manchester United fans are excited about Amorim's tactical approach #MUFC #NewEra #Positive",
        "description": "Sports content with hashtags about Manchester United and Amorim"
    }
]

print("Testing sentiment analysis API with sports-related content...")
print("This test verifies the fix for the 'positive_hashtags' attribute error")
print("-" * 80)

for i, data in enumerate(sports_test_data, 1):
    try:
        print(f"\nTest {i}: {data['description']}")
        print(f"Text: {data['text']}")
        
        # Send request to the API
        response = requests.post(url, json={"text": data['text']})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ SUCCESS - API returned status code 200")
            print(f"Sentiment: {result.get('sentiment')}")
            print(f"Prediction category: {result.get('prediction')}")
        else:
            print(f"✗ FAILED with status code {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\nFailed to connect to the Flask server at {url}")
        print("Make sure the server is running (python backend/sentiment_api.py)")
        break
    
    # Brief pause between requests
    time.sleep(1)

print("\nAll tests completed!")