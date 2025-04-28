from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib
import sys
import os
import torch
import datetime
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Add the src/models directory to the path to import from sentiment_analyzer
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'src', 'models'))
from sentiment_analyzer import SentimentModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up request logging
def log_request(route, req_data=None, status=None, error=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "route": route,
        "method": request.method,
        "remote_addr": request.remote_addr,
        "user_agent": request.headers.get('User-Agent', 'Unknown'),
        "status": status,
    }
    
    if req_data:
        # Truncate long text data for cleaner logs
        if 'text' in req_data and len(req_data['text']) > 100:
            truncated_data = req_data.copy()
            truncated_data['text'] = req_data['text'][:100] + "... [truncated]"
            log_entry["data"] = truncated_data
        else:
            log_entry["data"] = req_data
    
    if error:
        log_entry["error"] = str(error)
    
    print(f"REQUEST LOG: {json.dumps(log_entry, indent=2)}")
    
    # Also log to file for persistence
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"api_requests_{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
    
    with open(log_file, 'a') as f:
        f.write(f"{json.dumps(log_entry)}\n")

# Load the sentiment analysis model
try:
    sentiment_model = joblib.load('model.pkl')
    print("Sentiment model loaded successfully.")
except FileNotFoundError:
    try:
        # Try loading from src/models directory
        sentiment_model = joblib.load(os.path.join(parent_dir, 'src', 'models', 'model.pkl'))
        print("Sentiment model loaded from src/models directory.")
    except FileNotFoundError:
        sentiment_model = None
        print("Sentiment model file not found. Please ensure 'model.pkl' is available.")

# Initialize summarization model
summarization_model = None
summarization_tokenizer = None
summarizer = None

def load_summarization_model():
    global summarization_model, summarization_tokenizer, summarizer
    try:
        # Create cache directory
        cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using model cache directory: {cache_dir}")
        
        # Load the model for text summarization with local caching
        model_name = "facebook/bart-base"  # Changed from bart-large-cnn to bart-base
        print("Loading summarization tokenizer...")
        summarization_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        print("Loading summarization model...")
        summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        summarizer = pipeline(
            "summarization", 
            model=summarization_model, 
            tokenizer=summarization_tokenizer, 
            device=device,
            batch_size=1 if device == -1 else 8
        )
        print("✅ Summarization model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load summarization model: {e}")

@app.route('/')
def home():
    log_request('/', status="200 OK")
    return "Backend is running!"

def get_sentiment_percentage(category_index, raw_score=None):
    """
    Convert the sentiment category to a percentage value.
    
    Args:
        category_index: Integer from 0-4 representing sentiment category
        raw_score: Optional float between 0-1 from the model's raw prediction
        
    Returns:
        A percentage value from 0-100 representing sentiment intensity
    """
    # If we have a raw score from the model, use it for more precise percentage
    if raw_score is not None:
        # Scale the raw score (typically 0-1) to a percentage
        return round(raw_score * 100)
    
    # Otherwise map categories to percentage ranges
    category_to_percentage = {
        0: 10,   # Overwhelmingly negative: 10%
        1: 30,   # Negative: 30%
        2: 50,   # Neutral: 50%  
        3: 70,   # Positive: 70%
        4: 90    # Overwhelmingly positive: 90%
    }
    
    return category_to_percentage.get(category_index, 50)  # Default to 50% if unknown

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    log_request('/analyze', data, status="Processing")
    
    if not sentiment_model:
        error_msg = "Sentiment model not loaded. Please check the backend setup."
        log_request('/analyze', data, status="500 Error", error=error_msg)
        return jsonify({"error": error_msg}), 500

    if not data or 'text' not in data:
        error_msg = "Invalid input. Please provide 'text' in the request body."
        log_request('/analyze', data, status="400 Bad Request", error=error_msg)
        return jsonify({"error": error_msg}), 400

    text = data['text']
    
    try:
        # Get the sentiment category index (0-4) and corresponding label
        category_index = sentiment_model.predict([text])[0]
        sentiment_label = sentiment_model.get_sentiment_label(category_index)

        # Get the raw sentiment score if available
        raw_score = None
        try:
            raw_score = sentiment_model.get_raw_score(text)
        except (AttributeError, Exception) as e:
            print(f"Could not get raw score: {e}")

        # Calculate percentage based on category and raw score
        sentiment_percentage = get_sentiment_percentage(category_index, raw_score)
        # Fallback: ensure sentiment_percentage is always a valid number
        if sentiment_percentage is None or not isinstance(sentiment_percentage, (int, float)):
            sentiment_percentage = 50
        elif sentiment_percentage < 0 or sentiment_percentage > 100:
            sentiment_percentage = max(0, min(100, sentiment_percentage))

        # Add percentage to the label
        display_label = f"{sentiment_label} ({sentiment_percentage}%)"

        # Prepare the response with detailed sentiment information
        response = {
            "text": text, 
            "prediction": int(category_index),  # Convert to int for JSON serialization
            "sentiment": display_label,
            "sentiment_percentage": sentiment_percentage,
            "sentiment_category": {
                "0": "overwhelmingly negative",
                "1": "negative",
                "2": "neutral",
                "3": "positive", 
                "4": "overwhelmingly positive"
            }
        }
        
        log_request('/analyze', data, status="200 Success", error=None)
        return jsonify(response)
    except Exception as e:
        log_request('/analyze', data, status="500 Error", error=str(e))
        print(f"Error in sentiment analysis: {e}")
        return jsonify({"error": f"An error occurred during sentiment analysis: {str(e)}"}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    log_request('/summarize', data, status="Processing")
    
    if not summarizer:
        print("Summarizer not loaded yet, loading now...")
        load_summarization_model()
        if not summarizer:
            error_msg = "Summarization model not loaded. Please check the backend setup."
            log_request('/summarize', data, status="500 Error", error=error_msg)
            return jsonify({"error": error_msg}), 500

    if not data or 'text' not in data:
        error_msg = "Invalid input. Please provide 'text' in the request body."
        log_request('/summarize', data, status="400 Bad Request", error=error_msg)
        return jsonify({"error": error_msg}), 400

    text = data['text']
    sentiment_category = data.get('sentiment_category', 2)  # Default to neutral if not provided
    sentiment_label = data.get('sentiment_label', 'neutral')
    
    # Get optional parameters with defaults
    max_length = data.get('max_length', 60)  # Lower max_length for faster summarization
    min_length = data.get('min_length', 1)   # Remove minimum word cap
    
    # Trim very long inputs to improve performance
    max_input_length = 512  # Reduce input length for speed
    if len(text) > max_input_length:
        print(f"Trimming input text from {len(text)} to {max_input_length} characters")
        text = text[:max_input_length]
    
    try:
        start_time = datetime.datetime.now()
        # Generate the summary
        summary = summarizer(
            text, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False,
            num_beams=1,  # Use greedy decoding for speed
            early_stopping=True
        )
        
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Enhance the summary with model's opinion and sentiment reasoning
        opinion = generate_model_opinion(sentiment_category, sentiment_label)
        sentiment_reasoning = generate_sentiment_reasoning(text, sentiment_category, sentiment_label)
        enhanced_summary = (
            f"Summary: {summary[0]['summary_text']}\n\n"
            f"Model's Opinion: {opinion}\n"
            f"Reason: {sentiment_reasoning}"
        )
        
        response = {
            "original_text": text,
            "summary": enhanced_summary,
            "original_length": len(text),
            "summary_length": len(enhanced_summary),
            "processing_time_seconds": processing_time
        }
        
        log_request('/summarize', data, status=f"200 Success - Processed in {processing_time:.2f}s")
        return jsonify(response)
    except Exception as e:
        log_request('/summarize', data, status="500 Error", error=str(e))
        print(f"Error in text summarization: {e}")
        return jsonify({"error": f"An error occurred during summarization: {str(e)}"}), 500

def generate_sentiment_reasoning(text, sentiment_category, sentiment_label):
    """Generate reasoning for why the text has the assigned sentiment."""
    # Convert sentiment category to int if it's a string
    if isinstance(sentiment_category, str):
        try:
            sentiment_category = int(sentiment_category)
        except ValueError:
            sentiment_category = 2  # Default to neutral
    
    # Map of sentiment categories to reasoning templates
    reasoning_templates = {
        0: [  # Overwhelmingly negative
            "The text contains extremely negative language, accusations, and hostile rhetoric.",
            "There are strong negative statements that indicate severe criticism or anger.",
            "The content includes aggressive language that expresses deep dissatisfaction."
        ],
        1: [  # Negative
            "The language suggests dissatisfaction or disagreement.",
            "The text contains critical views and negative assessments.",
            "There are expressions of complaint or disapproval in the content."
        ],
        2: [  # Neutral
            "The text presents information in a balanced or factual manner.",
            "The content is primarily informative without strong emotional language.",
            "The statements are presented objectively without clear positive or negative bias."
        ],
        3: [  # Positive
            "The language includes supportive or approving statements.",
            "The text expresses satisfaction or agreement with the subject.",
            "There are positive assessments and constructive views in the content."
        ],
        4: [  # Overwhelmingly positive
            "The text contains extremely enthusiastic and celebratory language.",
            "There are strong expressions of praise, joy, or admiration.",
            "The content shows passionate approval and very positive sentiments."
        ]
    }
    
    # Choose a reasoning template based on sentiment category
    if sentiment_category in reasoning_templates:
        templates = reasoning_templates[sentiment_category]
        import random
        reasoning = random.choice(templates)
    else:
        reasoning = "The text contains mixed or ambiguous sentiment signals."
    
    # For social/political content, add more specific analysis
    political_keywords = ['war', 'soldiers', 'government', 'rights', 'policy', 'election', 'vote', 'democracy', 'freedom']
    if any(keyword in text.lower() for keyword in political_keywords):
        if sentiment_category <= 1:  # Negative sentiments
            reasoning += " The text discusses political or social issues with critical or concerned tone."
        elif sentiment_category >= 3:  # Positive sentiments
            reasoning += " The text discusses political or social issues with an optimistic or supportive perspective."
    
    return reasoning

def generate_model_opinion(sentiment_category, sentiment_label):
    """
    Generate a short opinion statement from the model based on the sentiment category and label.
    """
    opinions = {
        0: "This content is highly negative and may evoke strong negative emotions.",
        1: "This content is negative and expresses dissatisfaction or criticism.",
        2: "This content is neutral and presents information without strong emotion.",
        3: "This content is positive and expresses approval or satisfaction.",
        4: "This content is highly positive and conveys strong positive emotions."
    }
    # Try to use int category, fallback to neutral
    try:
        category = int(sentiment_category)
    except Exception:
        category = 2
    return opinions.get(category, "This content is neutral.")

if __name__ == '__main__':
    print("==================================================")
    print("Mood Map API Server Starting")
    print("Request logging enabled - monitoring for extension requests")
    print("==================================================")
    
    # Preload the summarization model at startup for faster first request
    print("Preloading summarization model...")
    load_summarization_model()
    
    # Start the server
    app.run(debug=False)  # Turn off debug mode for production