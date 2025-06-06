import torch
import torch.nn as nn
import joblib
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
from src.utils.logging_utils import setup_logging

# Setup logging
logger = setup_logging("logs/model_saver_logs.log")

# Define the model architecture to match what was used in training
class SentimentClassifier(nn.Module):
    """Neural network model for sentiment analysis using DistilBERT or BERT."""
    def __init__(self, model_type="distilbert", hidden_dim=768, output_dim=3):
        super(SentimentClassifier, self).__init__()
        self.model_type = model_type
        
        if model_type == "bert":
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # Freeze BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            # Freeze DistilBERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
            
        # Classification head with the same structure used in training
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            if self.model_type == "bert":
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

# Define the wrapper class at the module level to make it picklable
class SentimentAnalysisModelWrapper:
    def __init__(self, model, model_type="distilbert"):
        self.model = model
        self.model_type = model_type
        if model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def predict(self, texts):
        """Process texts and return sentiment predictions (0: Negative, 1: Positive)"""
        # Convert to list if single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize the texts
        encoded = self.tokenizer(
            texts, 
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            _, predictions = torch.max(outputs, 1)
        
        # Convert to list
        return predictions.tolist()
    
    def get_sentiment_label(self, category_index):
        """Convert a numeric sentiment category to a human-readable label."""
        labels = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        return labels.get(category_index, "unknown")
    
    def get_raw_score(self, text):
        """Return a raw confidence score for the sentiment prediction."""
        # Convert to list if single string
        if isinstance(text, str):
            text = [text]
        
        # Tokenize the text
        encoded = self.tokenizer(
            text, 
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get model predictions with probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Return the probability of the positive class (index 1)
        return probabilities[0][1].item()

def save_model(model, path):
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved successfully at {path}.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def create_and_save_model(output_path="model.pkl"):
    """Create a new model instance and save it to disk for the application."""
    print(f"Creating a sentiment analysis model...")
    
    # Create model instance
    model = SentimentClassifier()
    
    # Create wrapper
    wrapper_model = SentimentAnalysisModelWrapper(model)
    
    # Save using joblib
    try:
        save_model(wrapper_model, output_path)
        print(f"Model successfully saved to {output_path}")
        
        # Test the saved model
        print("Testing saved model with sample texts...")
        test_model = joblib.load(output_path)
        
        test_texts = [
            "I absolutely love this product, it's amazing!",
            "This is the worst experience I've ever had.",
            "The service was okay, nothing special."
        ]
        
        test_results = test_model.predict(test_texts)
        for text, result in zip(test_texts, test_results):
            sentiment = test_model.get_sentiment_label(result)
            print(f"'{text}' → {sentiment}")
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create and save the model
    create_and_save_model("model.pkl")
