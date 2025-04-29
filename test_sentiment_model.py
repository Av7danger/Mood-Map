import joblib
import sys
import os
import unittest
import pandas as pd
from src.models.sentiment_analyzer import analyze_sentiment
from create_robust_model import ensure_text_and_label_columns

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

class TestSentimentAnalyzer(unittest.TestCase):
    def test_hateful_content_detection(self):
        hateful_text = "This is a racist comment."
        result = analyze_sentiment(hateful_text)
        self.assertEqual(result["label"], "HATEFUL")
        self.assertEqual(result["score"], 1.0)

    def test_non_hateful_content(self):
        positive_text = "I love this product!"
        result = analyze_sentiment(positive_text)
        self.assertNotEqual(result["label"], "HATEFUL")

def test_ensure_text_and_label_columns():
    # Create a sample DataFrame to test the function
    data = pd.DataFrame({
        'Text': ['I love this!', 'This is bad.', 'It is okay.', 'Feeling great!'],
        'Sentiment': ['Positive', 'Negative', 'Neutral', 'Happy']
    })

    # Call the function to ensure proper columns
    updated_data = ensure_text_and_label_columns(data)

    # Assertions to verify the function's behavior
    assert 'text' in updated_data.columns, "The 'text' column was not created."
    assert 'label' in updated_data.columns, "The 'label' column was not created."
    assert updated_data['text'].equals(data['Text']), "The 'text' column does not match the 'Text' column."
    assert updated_data['label'].tolist() == [4, 0, 2, 4], "The 'label' column values are incorrect."

    print("All tests passed for ensure_text_and_label_columns!")

if __name__ == "__main__":
    unittest.main()
    test_ensure_text_and_label_columns()