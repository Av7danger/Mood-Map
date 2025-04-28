import joblib
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the sentiment model class from the correct location
from models.sentiment_analyzer import SentimentModel

# Define sentiment mapping
sentiment_categories = {
    0: "overwhelmingly negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "overwhelmingly positive"
}

# Try to load the model
print("Loading sentiment model...")
try:
    model = joblib.load('src/models/model.pkl')
    print("Model loaded successfully!")
    
    # Test some example texts
    test_texts = [
        "I absolutely love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special."
    ]
    
    # Get predictions
    print("\nTesting model predictions:")
    predictions = model.predict(test_texts)
    
    # Display results
    for text, pred in zip(test_texts, predictions):
        # Map numeric prediction to sentiment category
        sentiment = sentiment_categories.get(pred, "Unknown")
        
        # Display full information
        print(f"\nText: '{text}'")
        print(f"Prediction value: {pred}")
        print(f"Sentiment: {sentiment}")
        
        # Simplify to basic categories for easy verification
        if pred in [3, 4]:
            simple_sentiment = "Positive"
        elif pred in [0, 1]:
            simple_sentiment = "Negative"
        else:
            simple_sentiment = "Neutral"
        
        print(f"Simplified sentiment: {simple_sentiment}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()