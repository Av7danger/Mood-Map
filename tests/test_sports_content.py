import requests
import json
import time

# URL for the sentiment analysis endpoint
url = "http://127.0.0.1:5000/analyze"

# Added more sports-related test cases to cover a wider range of scenarios
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
    },
    {
        "text": "Manchester United's recent performance has been lackluster, and fans are losing hope.",
        "description": "Negative sports content about Manchester United's performance"
    },
    {
        "text": "Amorim's strategies have been praised by analysts, but execution remains a concern.",
        "description": "Mixed sports content about Amorim's strategies"
    },
    {
        "text": "The new signing for Manchester United is being hailed as a potential game-changer!",
        "description": "Positive sports content about a new signing for Manchester United"
    },
    {
        "text": "Manchester United's fanbase is divided over the recent managerial changes.",
        "description": "Neutral sports content about fan reactions to managerial changes"
    },
    {
        "text": "Amorim's tenure at Manchester United could redefine the club's legacy.",
        "description": "Optimistic sports content about Amorim's potential impact"
    },
    {
        "text": "The tactical decisions in the last match were questionable, leading to a disappointing loss.",
        "description": "Critical sports content about tactical decisions"
    },
    {
        "text": "Manchester United's victory last night was a testament to their resilience and teamwork!",
        "description": "Overwhelmingly positive sports content about a recent victory"
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