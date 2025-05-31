#!/usr/bin/env python
"""
Script to train an improved sentiment analysis model.
This uses scikit-learn with TF-IDF features and a pipeline approach
for more accurate sentiment classification.
"""
import os
import sys
import time
import pickle
import re  # Add missing import
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup output paths
output_dir = project_root
model_path = os.path.join(output_dir, "improved_model.pkl")
log_path = os.path.join(project_root, "logs", f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Ensure logs directory exists
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

# Configure simple logging to file
def log(message):
    with open(log_path, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

log(f"Starting improved model training script")
log(f"Model will be saved to: {model_path}")

# Try to import required libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    log("Successfully imported scikit-learn libraries")
except ImportError as e:
    log(f"Error importing scikit-learn: {e}")
    log("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn>=1.0.0", "pandas>=1.3.0", "nltk>=3.6.0"])
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    log("Successfully installed and imported scikit-learn libraries")

# Import NLTK for text preprocessing
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    # Make sure punkt is properly installed for English
    try:
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
        test = tokenizer.tokenize("This is a test. This is another test.")
    except LookupError:
        nltk.download('punkt', quiet=False)
    except Exception as e:
        print(f"Warning: {e}")
        # Fallback to manual download
        nltk.download('punkt', quiet=False)
    
    log("Successfully imported and set up NLTK")
except ImportError:
    log("Error importing NLTK. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk>=3.6.0"])
    
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    log("Successfully installed and set up NLTK")

class ImprovedSentimentModel:
    """
    Improved sentiment analysis model with better preprocessing,
    feature extraction, and ensemble classification.
    """
    def __init__(self):
        self.pipeline = None
        self.classes_ = [0, 1, 2]  # negative, neutral, positive
        self.class_names = ["negative", "neutral", "positive"]
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Sentiment lexicons
        self.positive_words = set([
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
            "terrific", "outstanding", "superb", "brilliant", "awesome", 
            "fabulous", "incredible", "love", "best", "perfect", "happy",
            "enjoy", "like", "glad", "pleasant", "satisfied", "impressive",
            "exceptional", "magnificent", "marvelous", "delightful", "positive"
        ])
        
        self.negative_words = set([
            "bad", "terrible", "awful", "horrible", "worst", "poor", "dreadful",
            "disappointing", "mediocre", "inferior", "hate", "dislike", "horrible",
            "annoying", "frustrated", "useless", "waste", "sad", "angry", "broken",
            "disappointed", "unpleasant", "negative", "fail", "problem", "issue",
            "error", "disappointing", "defective", "sucks", "pathetic", "failure"
        ])
        
        self.intensifiers = set([
            "very", "extremely", "absolutely", "completely", "totally", "utterly",
            "highly", "incredibly", "remarkably", "exceptionally", "hugely", "immensely",
            "seriously", "really", "truly", "particularly", "especially", "decidedly",
            "greatly", "intensely", "extraordinarily", "unusually", "exceedingly"
        ])
        
        self.negations = set([
            "not", "no", "never", "neither", "none", "nobody", "nowhere", "nothing",
            "nor", "hardly", "barely", "scarcely", "doesn't", "isn't", "wasn't",
            "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"
        ])
    
    def preprocess_text(self, text):
        """Advanced text preprocessing with lemmatization and feature enrichment"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stop words
        processed_tokens = []
        has_negation = False
        
        for i, token in enumerate(tokens):
            # Check for negations that might flip sentiment
            if token in self.negations:
                has_negation = True
            
            # Skip stop words except negations
            if token in self.stop_words and token not in self.negations:
                continue
                
            # Lemmatize
            lemma = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)
            
            # Add special sentiment markers based on lexicons
            if lemma in self.positive_words:
                if has_negation and i > 0 and tokens[i-1] in self.negations:
                    processed_tokens.append("NEG_POSITIVE")
                else:
                    processed_tokens.append("POSITIVE_WORD")
                    
            if lemma in self.negative_words:
                if has_negation and i > 0 and tokens[i-1] in self.negations:
                    processed_tokens.append("NEG_NEGATIVE")
                else:
                    processed_tokens.append("NEGATIVE_WORD")
        
        # Add special feature tokens based on patterns
        if "!" in text:
            processed_tokens.append("HAS_EXCLAMATION")
            
        if "?" in text:
            processed_tokens.append("HAS_QUESTION")
            
        # Count uppercase words (potential emphasis)
        uppercase_word_count = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        if uppercase_word_count > 0:
            processed_tokens.append("HAS_EMPHASIS")
        
        # Detect mixed sentiment (both positive and negative words)
        has_positive = any(token in self.positive_words for token in tokens)
        has_negative = any(token in self.negative_words for token in tokens)
        if has_positive and has_negative:
            processed_tokens.append("MIXED_SENTIMENT")
        
        # Detect hedging/uncertainty language
        uncertainty_words = ["maybe", "perhaps", "possibly", "might", "could", "uncertain"]
        if any(word in tokens for word in uncertainty_words):
            processed_tokens.append("UNCERTAINTY")
        
        return " ".join(processed_tokens)
    
    def extract_additional_features(self, text):
        """Extract additional features beyond TF-IDF"""
        features = {}
        
        if not isinstance(text, str):
            # Return default features for non-string input
            features["text_length"] = 0
            features["avg_word_length"] = 0
            features["pos_word_count"] = 0
            features["neg_word_count"] = 0
            features["exclamation_count"] = 0
            features["question_count"] = 0
            features["uppercase_ratio"] = 0
            return features
        
        # Text length features
        features["text_length"] = len(text)
        
        words = text.split()
        if words:
            features["avg_word_length"] = sum(len(word) for word in words) / len(words)
        else:
            features["avg_word_length"] = 0
        
        # Sentiment word counts
        lower_text = text.lower()
        tokens = word_tokenize(lower_text)
        
        # Count positive and negative words
        pos_word_count = sum(1 for token in tokens if token in self.positive_words)
        neg_word_count = sum(1 for token in tokens if token in self.negative_words)
        
        features["pos_word_count"] = pos_word_count
        features["neg_word_count"] = neg_word_count
        
        # Punctuation features
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        
        # Stylistic features
        uppercase_chars = sum(1 for c in text if c.isupper())
        features["uppercase_ratio"] = uppercase_chars / len(text) if len(text) > 0 else 0
        
        # Emotional intensity features
        features["intensifier_count"] = sum(1 for token in tokens if token in self.intensifiers)
        
        # Negation features
        features["negation_count"] = sum(1 for token in tokens if token in self.negations)
        
        return features
    
    def fit(self, X, y):
        """Train the model on the provided data"""
        log(f"Starting model training with {len(X)} examples")
        
        # Create a pipeline with TF-IDF and classifier
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                preprocessor=self.preprocess_text,
                ngram_range=(1, 3),
                max_features=10000,
                min_df=2,
                use_idf=True
            )),
            ('classifier', CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight='balanced', dual=False),
                cv=3
            ))
        ])
        
        # Train the model
        start_time = time.time()
        self.pipeline.fit(X, y)
        training_time = time.time() - start_time
        
        log(f"Model training completed in {training_time:.2f} seconds")
        
        # Store class information
        self.classes_ = self.pipeline.named_steps['classifier'].classes_
        
        return self
    
    def predict(self, X):
        """Predict sentiment class for the given text"""
        if not isinstance(X, list):
            X = [X]
            
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call fit() first.")
            
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict probability of each sentiment class"""
        if not isinstance(X, list):
            X = [X]
            
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call fit() first.")
            
        return self.pipeline.predict_proba(X)
    
    def get_sentiment_label(self, prediction):
        """Convert numeric prediction to text label"""
        if prediction == 0:
            return "negative"
        elif prediction == 1:
            return "neutral"
        elif prediction == 2:
            return "positive"
        else:
            return "unknown"
    
    def analyze_sentiment(self, text):
        """Full sentiment analysis with explanation"""
        if not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "score": 1.0,
                "confidence": 0.33,
                "explanation": "Invalid input"
            }
            
        prediction = self.predict([text])[0]
        
        # Get probabilities for confidence score
        proba = self.predict_proba([text])[0]
        confidence = float(proba[prediction])
        
        # Get label
        sentiment = self.get_sentiment_label(prediction)
        
        # Generate explanation
        features = self.extract_additional_features(text)
        explanation = self._generate_explanation(text, sentiment, features, confidence)
        
        return {
            "sentiment": sentiment,
            "score": float(prediction),
            "category": int(prediction),
            "confidence": confidence,
            "explanation": explanation
        }
    
    def _generate_explanation(self, text, sentiment, features, confidence):
        """Generate a human-readable explanation for the sentiment prediction"""
        explanation = []
        
        # Comment on confidence
        if confidence > 0.8:
            explanation.append(f"High confidence ({confidence:.2f}) {sentiment} sentiment detected.")
        elif confidence > 0.6:
            explanation.append(f"Moderate confidence ({confidence:.2f}) that this is {sentiment}.")
        else:
            explanation.append(f"Low confidence ({confidence:.2f}) prediction of {sentiment} sentiment.")
            
        # Explain based on text features
        if features["pos_word_count"] > 0 and features["neg_word_count"] > 0:
            explanation.append("Text contains both positive and negative words, suggesting mixed sentiment.")
        elif features["pos_word_count"] > features["neg_word_count"]:
            explanation.append(f"Contains {features['pos_word_count']} positive words.")
        elif features["neg_word_count"] > features["pos_word_count"]:
            explanation.append(f"Contains {features['neg_word_count']} negative words.")
            
        # Comment on stylistic elements
        if features["exclamation_count"] > 1:
            explanation.append("Multiple exclamation marks suggest emotional intensity.")
        if features["question_count"] > 1:
            explanation.append("Contains questions, possibly indicating uncertainty.")
        if features["uppercase_ratio"] > 0.2:
            explanation.append("Uppercase letters suggest emphasis or strong emotion.")
            
        # Note negations if present
        if features["negation_count"] > 0:
            explanation.append("Contains negations which may reverse sentiment.")
            
        return " ".join(explanation)

def load_and_prepare_data():
    """Load and prepare training data for the sentiment model"""
    log("Loading and preparing training data")
    
    # First try to find sentiment datasets in the project
    possible_data_paths = [
        os.path.join(project_root, "data", "processed", "sentiment_dataset.csv"),
        os.path.join(project_root, "data", "raw", "sentiment_data.csv"),
        os.path.join(project_root, "data", "twitter_sentiment.csv")
    ]
    
    df = None
    
    # Try to load from existing files
    for path in possible_data_paths:
        if os.path.exists(path):
            log(f"Found dataset at {path}")
            try:
                df = pd.read_csv(path)
                log(f"Loaded {len(df)} examples from {path}")
                break
            except Exception as e:
                log(f"Error loading {path}: {e}")
    
    # If no dataset found, create a synthetic one
    if df is None:
        log("No dataset found, creating synthetic training data")
        
        # Create synthetic dataset with balanced classes
        texts = []
        labels = []
        
        # Positive examples
        positive_examples = [
            "I love this product! It's amazing and works perfectly.",
            "This is awesome, best purchase I've ever made.",
            "Excellent service and quality, highly recommended!",
            "I'm very happy with my purchase, it exceeded my expectations.",
            "This app is fantastic, it has improved my productivity.",
            "Great experience, will definitely use again!",
            "The customer service was outstanding and very helpful.",
            "This is exactly what I needed, works like a charm.",
            "I'm impressed with the quality and attention to detail.",
            "Wonderful product, easy to use and very effective.",
            "This is so good, I can't believe I didn't buy it sooner.",
            "Absolutely love it! Perfect in every way.",
            "Very satisfied with my experience, thank you!",
            "Impressive performance, exceeded all my expectations.",
            "I'm really enjoying this, it's made such a difference."
        ]
        
        # Neutral examples
        neutral_examples = [
            "The package arrived on time and contained all items.",
            "Product works as described, nothing special.",
            "It's okay, serves its purpose but nothing extraordinary.",
            "Average quality, does what it's supposed to do.",
            "Neutral experience, neither good nor bad.",
            "Delivery was on time, product is as expected.",
            "It works fine, but I'm not particularly impressed.",
            "The product is functional but basic.",
            "It's alright, nothing to complain about.",
            "Standard service, met my basic expectations.",
            "Not bad, not great, just average.",
            "It's fine for the price, about what I expected.",
            "Received the item as described, no issues.",
            "The product is acceptable, does the job.",
            "Middle of the road experience, neither impressed nor disappointed."
        ]
        
        # Negative examples
        negative_examples = [
            "This is the worst experience ever. I hate it.",
            "Terrible product, doesn't work as advertised.",
            "Extremely disappointed with the quality, would not recommend.",
            "Poor service, will not be using again.",
            "Don't waste your money on this, it's awful.",
            "Very frustrated with this purchase, complete waste of time.",
            "This is defective and the company won't help.",
            "Horrible customer service, they didn't resolve my issue.",
            "The product broke after one use, very poor quality.",
            "I regret buying this, it's nothing like described.",
            "Awful experience from start to finish.",
            "This is a complete scam, stay away!",
            "Extremely unhappy with this purchase.",
            "The worst service I've ever experienced.",
            "This product is garbage, save your money."
        ]
        
        # Mixed sentiment examples
        mixed_examples = [
            "This movie was great but the ending could have been better.",
            "Good product overall, though the price is a bit high.",
            "The food was delicious, but the service was slow.",
            "I like the features, but the interface is confusing.",
            "Great concept but poor execution.",
            "It has some good qualities but also significant flaws.",
            "The performance is excellent, however battery life is disappointing.",
            "Love the design, hate the functionality.",
            "Started great but ended terribly.",
            "Good product but shipping was a nightmare.",
            "Nice features but too expensive for what you get.",
            "The staff was friendly but not very knowledgeable.",
            "It works well most of the time, but crashes occasionally.",
            "Beautiful design but uncomfortable to use for long periods.",
            "Some parts are excellent while others are terrible."
        ]
        
        # Add positive examples (label 2)
        texts.extend(positive_examples)
        labels.extend([2] * len(positive_examples))
        
        # Add neutral examples (label 1)
        texts.extend(neutral_examples)
        labels.extend([1] * len(neutral_examples))
        
        # Add negative examples (label 0)
        texts.extend(negative_examples)
        labels.extend([0] * len(negative_examples))
        
        # Add mixed examples (distribute between neutral and the sentiment that's strongest)
        for example in mixed_examples:
            if "but" in example.lower() or "however" in example.lower() or "though" in example.lower():
                # Check which part is emphasized more
                parts = re.split(r'\sbut\s|\showever\s|\sthough\s', example.lower())
                if len(parts) >= 2:
                    first_part = parts[0]
                    second_part = parts[1]
                    
                    # Count sentiment words in each part
                    model = ImprovedSentimentModel()
                    pos_first = sum(1 for word in first_part.split() if word in model.positive_words)
                    neg_first = sum(1 for word in first_part.split() if word in model.negative_words)
                    pos_second = sum(1 for word in second_part.split() if word in model.positive_words)
                    neg_second = sum(1 for word in second_part.split() if word in model.negative_words)
                    
                    first_sentiment = 2 if pos_first > neg_first else 0 if neg_first > pos_first else 1
                    second_sentiment = 2 if pos_second > neg_second else 0 if neg_second > pos_second else 1
                    
                    # If the second part is emphasized (recency effect)
                    if len(second_part) > len(first_part):
                        texts.append(example)
                        labels.append(second_sentiment)
                    else:
                        texts.append(example)
                        labels.append(first_sentiment)
                else:
                    # Default to neutral for mixed sentiment that can't be parsed
                    texts.append(example)
                    labels.append(1)
            else:
                # Default to neutral for mixed sentiment
                texts.append(example)
                labels.append(1)
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'label': labels})
        log(f"Created synthetic dataset with {len(df)} examples")
        
        # Save the synthetic dataset for future use
        df_path = os.path.join(project_root, "data", "synthetic_sentiment_dataset.csv")
        os.makedirs(os.path.dirname(df_path), exist_ok=True)
        df.to_csv(df_path, index=False)
        log(f"Saved synthetic dataset to {df_path}")
    
    return df

def get_balanced_sample(df, target_column='label', n_per_class=None):
    """Get a balanced sample dataset with equal representation of each class"""
    if n_per_class is None:
        # If not specified, use the min class count
        n_per_class = df[target_column].value_counts().min()
    
    # Get samples for each class
    balanced_df = pd.DataFrame()
    for class_val in df[target_column].unique():
        class_df = df[df[target_column] == class_val]
        if len(class_df) > n_per_class:
            sampled = class_df.sample(n=n_per_class, random_state=42)
        else:
            sampled = class_df
        balanced_df = pd.concat([balanced_df, sampled])
    
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

def main():
    """Main function to train and save the improved sentiment model"""
    start_time = time.time()
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get column names (may vary depending on the dataset)
    text_column = 'text' if 'text' in df.columns else 'content' if 'content' in df.columns else df.columns[0]
    label_column = 'label' if 'label' in df.columns else 'sentiment' if 'sentiment' in df.columns else df.columns[1]
    
    log(f"Using '{text_column}' as text column and '{label_column}' as label column")
    
    # Make sure we have numeric labels
    if df[label_column].dtype == 'object':
        # If labels are text, convert to numeric
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df[label_column] = df[label_column].map(lambda x: label_map.get(str(x).lower(), 1))
        log("Converted text labels to numeric values")
    
    # Balance the dataset
    balanced_df = get_balanced_sample(df, target_column=label_column)
    log(f"Balanced dataset has {len(balanced_df)} examples, {balanced_df[label_column].value_counts().to_dict()}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df[text_column], 
        balanced_df[label_column],
        test_size=0.2,
        random_state=42,
        stratify=balanced_df[label_column]
    )
    
    log(f"Split into {len(X_train)} training examples and {len(X_test)} test examples")
    
    # Train the model
    model = ImprovedSentimentModel()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log(f"Model accuracy: {accuracy:.4f}")
    
    # Get a detailed classification report
    report = classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"])
    log(f"Classification report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log(f"Confusion matrix:\n{cm}")
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    log(f"Model saved to {model_path}")
    
    # Sample predictions
    log("Sample predictions:")
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience ever. I hate it.",
        "The package arrived on time and contained all items.",
        "This movie was great but the ending could have been better.",
        "I'm not sure if I like this or not."
    ]
    
    for text in test_texts:
        result = model.analyze_sentiment(text)
        log(f"Text: '{text}'")
        log(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        log(f"Explanation: {result['explanation']}")
        log("---")
    
    # Final timing
    total_time = time.time() - start_time
    log(f"Total training and evaluation time: {total_time:.2f} seconds")
    log("Improved sentiment model training completed successfully")
    
    # Create a backup of the old model if it exists
    original_model_path = os.path.join(project_root, "model.pkl")
    if os.path.exists(original_model_path):
        backup_path = os.path.join(project_root, "model.pkl.backup")
        try:
            import shutil
            shutil.copy2(original_model_path, backup_path)
            log(f"Created backup of original model at {backup_path}")
        except Exception as e:
            log(f"Warning: Could not create backup of original model: {e}")
    
    # Copy the improved model to replace the original model.pkl
    try:
        import shutil
        shutil.copy2(model_path, original_model_path)
        log(f"Replaced original model.pkl with improved model")
        
        # Also copy to backend folder
        backend_model_path = os.path.join(project_root, "backend", "model.pkl")
        shutil.copy2(model_path, backend_model_path)
        log(f"Copied improved model to backend folder")
    except Exception as e:
        log(f"Warning: Could not replace original model: {e}")

if __name__ == "__main__":
    main()