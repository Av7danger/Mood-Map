import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import requests
import json
import time
import unittest
from src.utils.input_validation import validate_input
from src.utils.logging_utils import setup_logging

# Setup logging for tests
logger = setup_logging("test_logs.log")

# Allow more time for Flask server to start and load models
print("Waiting for Flask server to initialize (15 seconds)...")
time.sleep(15)  # Increased wait time from 5 to 15 seconds

# URL for the sentiment analysis endpoint
url = "http://127.0.0.1:5000/analyze"

# Test sentences with expected sentiments
test_data = [
    {"text": "I absolutely love this product, it's amazing!", "expected": "Positive"},
    {"text": "This is the worst experience I've ever had.", "expected": "Negative"},
    {"text": "The service was okay, nothing special.", "expected": "Neutral/Negative"}
]

# Sentiment category mapping
sentiment_categories = {
    0: "overwhelmingly negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "overwhelmingly positive"
}

class TestSentimentAPI(unittest.TestCase):
    def test_validate_input(self):
        valid_data = {"text": "This is a test."}
        invalid_data = {"content": "Missing 'text' key."}

        self.assertTrue(validate_input(valid_data))
        self.assertFalse(validate_input(invalid_data))

        logger.info("Input validation tests passed.")

    def test_sentiment_analysis(self):
        # Send requests to the API
        print("Testing sentiment analysis API...")
        for i, data in enumerate(test_data, 1):
            try:
                response = requests.post(url, json={"text": data["text"]})
                
                if response.status_code == 200:
                    result = response.json()
                    prediction_num = result.get("prediction")
                    sentiment_text = sentiment_categories.get(prediction_num, "Unknown")
                    
                    # Simplify to match expected categories
                    if sentiment_text in ["positive", "overwhelmingly positive"]:
                        simplified_prediction = "Positive"
                    elif sentiment_text in ["negative", "overwhelmingly negative"]:
                        simplified_prediction = "Negative"
                    else:
                        simplified_prediction = "Neutral"
                    
                    print(f"\nTest {i}:")
                    print(f"Text: {data['text']}")
                    print(f"Expected: {data['expected']}")
                    print(f"Predicted: {simplified_prediction}")
                    print(f"Full API response: {json.dumps(result, indent=2)}")
                    
                    # Check if prediction matches expectation
                    if (simplified_prediction == data["expected"]) or (data["expected"] == "Neutral/Negative" and simplified_prediction in ["Neutral", "Negative"]):
                        print("✓ PASSED")
                    else:
                        print("✗ FAILED")
                else:
                    print(f"\nTest {i} failed with status code {response.status_code}")
                    print(f"Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                print(f"\nFailed to connect to the Flask server at {url}")
                print("Make sure the server is running (cd backend && python sentiment_api.py)")
                break
            
            # Brief pause between requests
            time.sleep(1)

if __name__ == "__main__":
    unittest.main()