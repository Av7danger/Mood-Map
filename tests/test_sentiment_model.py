import joblib
import sys
import os
import unittest
import pandas as pd
import pytest
from src.models.sentiment_analyzer import analyze_sentiment, SentimentModel
from scripts.create_robust_model import ensure_text_and_label_columns
from src.utils.data_validation import validate_tweets
from src.utils.logging_utils import setup_logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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

# Setup logging for tests
logger = setup_logging("test_logs.log")

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

    def test_validate_tweets(self):
        tweets = [
            ["1", "2025-04-30", "NO_QUERY", "user1", "This is a valid tweet.", 10, 5],
            ["2", "2025-04-30", "NO_QUERY", "user2", "", 0, 0]  # Invalid tweet
        ]

        valid_tweets = validate_tweets(tweets)
        self.assertEqual(len(valid_tweets), 1)
        logger.info("Tweet validation tests passed.")

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

@pytest.mark.parametrize("text, expected_category", [
    ("This is absolutely amazing! Best experience of my life!", 4),  # Overwhelmingly positive
    ("I had a really good time and enjoyed the experience.", 3),      # Positive
    ("It was okay, nothing special.", 2),                            # Neutral
    ("I didn't really enjoy this and wouldn't recommend it.", 1),    # Negative
    ("Terrible experience! Complete waste of money and time!", 0),   # Overwhelmingly negative
    ("While I loved the service, the product itself was disappointing.", 2), # Mixed sentiment
    ("The absolute worst product I've ever used, incredible results!", 0),   # Sarcasm detection
    ("The food was decent, but the service was exceptional.", 3),    # Positive with mixed context
    ("I can't believe how bad this was. Never again!", 0),           # Overwhelmingly negative
    ("The staff was friendly, but the product quality was poor.", 1) # Negative with mixed context
])
def test_sentiment_prediction(text, expected_category):
    prediction = model.predict([text])[0]
    assert prediction == expected_category, f"Expected {expected_category}, got {prediction}"

# Additional test examples for sentiment analysis
@pytest.mark.parametrize("text, expected_category", [
    # Overwhelmingly Positive Examples
    ("This is the best day of my life!", 4),
    ("Absolutely phenomenal! Exceeded all expectations!", 4),
    ("Outstanding quality and service!", 4),
    ("I am thrilled with the results!", 4),
    ("This product is a game-changer!", 4),
    ("I couldn't be happier with this purchase!", 4),
    ("This is a masterpiece of design and functionality!", 4),
    # Positive Examples
    ("I enjoyed using this product, it made my work easier.", 3),
    ("The service was good and the staff was helpful.", 3),
    ("Overall a pleasant experience that I would recommend.", 3),
    ("The food was delicious and the ambiance was great.", 3),
    ("I am happy with my purchase.", 3),
    ("The product met my expectations and worked well.", 3),
    ("I would definitely buy this again.", 3),
    # Neutral Examples
    ("It was okay, nothing special.", 2),
    ("The performance was acceptable, but not memorable.", 2),
    ("It works as expected, nothing more, nothing less.", 2),
    ("The product is decent for its price.", 2),
    ("I have mixed feelings about this.", 2),
    ("Neither good nor bad, just average.", 2),
    ("It does the job, but nothing extraordinary.", 2),
    # Negative Examples
    ("I was disappointed with how this turned out.", 1),
    ("There were several issues that made this experience unpleasant.", 1),
    ("Not what I expected, and I feel let down.", 1),
    ("The quality of the product is subpar.", 1),
    ("I regret buying this.", 1),
    ("The experience was below average and frustrating.", 1),
    ("I wouldn't recommend this to anyone.", 1),
    # Overwhelmingly Negative Examples
    ("This is the worst experience I've ever had!", 0),
    ("Absolutely terrible! A complete waste of money!", 0),
    ("I hate this product, it's awful!", 0),
    ("This is a disaster, I will never use this again!", 0),
    ("Horrible experience, avoid at all costs!", 0),
    ("This is the most disappointing purchase I've ever made!", 0),
    ("I can't believe how bad this is, it's shocking!", 0)
])
def test_sentiment_prediction_extended(text, expected_category):
    prediction = model.predict([text])[0]
    assert prediction == expected_category, f"Expected {expected_category}, got {prediction}"

# Script to calculate accuracy
def calculate_accuracy():
    test_cases = [
        ("This is absolutely amazing! Best experience of my life!", 4),
        ("I had a really good time and enjoyed the experience.", 3),
        ("It was okay, nothing special.", 2),
        ("I didn't really enjoy this and wouldn't recommend it.", 1),
        ("Terrible experience! Complete waste of money and time!", 0),
        ("While I loved the service, the product itself was disappointing.", 2),
        ("The absolute worst product I've ever used, incredible results!", 0),
        ("The food was decent, but the service was exceptional.", 3),
        ("I can't believe how bad this was. Never again!", 0),
        ("The staff was friendly, but the product quality was poor.", 1)
    ]

    correct_predictions = 0
    for text, expected_category in test_cases:
        prediction = model.predict([text])[0]
        if prediction == expected_category:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_cases) * 100
    print(f"Model Accuracy: {accuracy:.2f}%")

# Script to calculate overall accuracy on a dataset
def calculate_overall_accuracy():
    # Load the test dataset (replace with the actual dataset path if available)
    dataset_path = "data/raw/sentimentdataset.csv"
    try:
        # Load the dataset into a DataFrame
        data = pd.read_csv(dataset_path, encoding="ISO-8859-1", header=None)
        data.columns = ["target", "id", "date", "flag", "user", "text"]

        # Map target values to sentiment categories (adjust mapping if needed)
        target_mapping = {
            0: 0,  # Overwhelmingly Negative
            1: 1,  # Negative
            2: 2,  # Neutral
            3: 3,  # Positive
            4: 4   # Overwhelmingly Positive
        }
        data["mapped_target"] = data["target"].map(target_mapping)

        # Filter out rows with unmapped targets
        data = data.dropna(subset=["mapped_target"])

        # Get the text and labels
        texts = data["text"].tolist()
        labels = data["mapped_target"].astype(int).tolist()

        # Make predictions
        predictions = model.predict(texts)

        # Calculate accuracy
        correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        accuracy = correct_predictions / len(labels) * 100

        print(f"Overall Model Accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"Error calculating overall accuracy: {e}")

if __name__ == "__main__":
    unittest.main()
    test_ensure_text_and_label_columns()
    calculate_accuracy()
    calculate_overall_accuracy()