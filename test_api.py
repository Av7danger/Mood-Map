import requests
import json
import sys

def test_sentiment_api():
    """
    Test the sentiment analysis API by sending a few sample texts.
    """
    print("Testing Sentiment Analysis API...")
    print("================================")
    
    # API endpoint
    url = "https://localhost:5000/analyze"
    
    # Sample texts to test with different sentiments
    test_texts = [
        "I absolutely love this product! It's the best thing ever.",
        "This is terrible, I'm very disappointed with the service.",
        "The weather today is partly cloudy with some sun.",
        "I'm so excited for the upcoming vacation!",
        "I'm feeling quite angry about how this was handled."
    ]
    
    # Test each text
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:30]}...'")
        
        try:
            # Make the API request with SSL verification disabled 
            # (since we're using self-signed certificates)
            response = requests.post(
                url, 
                json={"text": text}, 
                verify=False  # Disable SSL verification for self-signed certs
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success! Sentiment: {result['sentiment']}")
                print(f"  Prediction category: {result['prediction']}")
                print(f"  Sentiment percentage: {result['sentiment_percentage']}%")
            else:
                print(f"✗ Error: Status code {response.status_code}")
                print(f"  Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error: {e}")
            print("  Make sure the API server is running at https://localhost:5000")
            
    print("\n================================")
    print("API testing complete!")

if __name__ == "__main__":
    # Suppress SSL warnings for cleaner output
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        test_sentiment_api()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)