from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib
import sys
import os
import torch
import datetime
import json
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Fix the import paths by properly adding the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now import from src after fixing the path
from src.utils.input_validation import validate_input
from src.models.sentiment_analyzer import analyze_sentiment

# Add the src/models directory to the path to import from sentiment_analyzer
sys.path.append(os.path.join(parent_dir, 'src', 'models'))
from sentiment_analyzer import SentimentModel
from src.training.train_sentiment_model import SentimentClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(
    filename="backend_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "sentiment_api_config.json")
if not os.path.exists(config_path):
    # Try to copy from config directory if not found in backend
    src_config = os.path.join(parent_dir, "config", "sentiment_api_config.json")
    if os.path.exists(src_config):
        import shutil
        print(f"Copying config from {src_config} to {config_path}")
        shutil.copy2(src_config, config_path)
    else:
        print("Warning: Could not find sentiment_api_config.json. Using default configuration.")
        config = {
            "positive_keywords": ["love", "happy", "great", "excellent", "good", "best", "amazing"],
            "negative_keywords": ["hate", "bad", "terrible", "awful", "worst", "horrible", "disappointing"]
        }

if os.path.exists(config_path):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    print("Configuration loaded successfully.")
else:
    print("Using default configuration.")
    config = {
        "positive_keywords": ["love", "happy", "great", "excellent", "good", "best", "amazing"],
        "negative_keywords": ["hate", "bad", "terrible", "awful", "worst", "horrible", "disappointing"]
    }

# Load the sentiment analysis model
try:
    sentiment_model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))
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
    Convert the sentiment category to a percentage value with improved granularity.
    
    Args:
        category_index: Integer from 0-4 representing sentiment category
        raw_score: Optional float between 0-1 from the model's raw prediction
        
    Returns:
        A percentage value from 0-100 representing sentiment intensity
    """
    # If we have a raw score from the model, use it for more precise percentage
    if raw_score is not None:
        # Scale the raw score (typically 0-1) to a percentage with bias correction
        # This improves the distribution of sentiment percentages to avoid clustering
        if raw_score < 0.5:
            # Expand the lower half of the scale (0-0.5) to (0-40)
            return round(raw_score * 80)
        else:
            # Expand the upper half of the scale (0.5-1) to (40-100)
            return round(40 + (raw_score - 0.5) * 120)
    
    # Otherwise use an improved mapping for categories with better granularity
    category_to_percentage = {
        0: 10,   # Overwhelmingly negative: 10%
        1: 30,   # Negative: 30%
        2: 50,   # Neutral: 50%  
        3: 70,   # Positive: 70%
        4: 90    # Overwhelmingly positive: 90%
    }
    
    return category_to_percentage.get(category_index, 50)  # Default to 50% if unknown

def get_sentiment_emojis(category_index, sentiment_percentage=None):
    """
    Generate appropriate emojis based on sentiment category and intensity.
    
    Args:
        category_index: Integer from 0-4 representing sentiment category
        sentiment_percentage: Optional percentage value for more granular emoji selection
        
    Returns:
        Dict containing primary and secondary emojis representing the sentiment
    """
    # Define emoji mappings for each sentiment category with intensity variations
    emoji_mappings = {
        0: {  # Overwhelmingly negative
            "primary": ["😠", "😡", "🤬", "😤"],
            "secondary": ["💔", "👎", "🚫", "❌", "😖"]
        },
        1: {  # Negative
            "primary": ["🙁", "😕", "😒", "😞"],
            "secondary": ["👎", "💢", "🤦", "🙄"]
        },
        2: {  # Neutral
            "primary": ["😐", "🤔", "😶", "🫤"],
            "secondary": ["⚖️", "🤷", "📊", "📝"]
        },
        3: {  # Positive
            "primary": ["🙂", "😊", "👍", "😀"],
            "secondary": ["👌", "✅", "💯", "🔆"]
        },
        4: {  # Overwhelmingly positive
            "primary": ["😁", "🤩", "😍", "🥰"],
            "secondary": ["🎉", "🎊", "⭐", "💖", "🔥"]
        }
    }
    
    # Default to middle intensity if no percentage provided
    if sentiment_percentage is None:
        primary_emoji = emoji_mappings[category_index]["primary"][1]
        secondary_emoji = emoji_mappings[category_index]["secondary"][1]
        return {"primary": primary_emoji, "secondary": secondary_emoji}
    
    # Use percentage to determine intensity within the category
    primary_emojis = emoji_mappings[category_index]["primary"]
    secondary_emojis = emoji_mappings[category_index]["secondary"]
    
    # Map percentage to emoji index
    if category_index in [0, 1]:  # Negative categories
        # For negative emotions, higher percentage means more intense negative
        index = min(len(primary_emojis) - 1, int(sentiment_percentage / 25))
    elif category_index in [3, 4]:  # Positive categories
        # For positive emotions, higher percentage means more intense positive
        index = min(len(primary_emojis) - 1, int(sentiment_percentage / 25))
    else:  # Neutral category
        # For neutral, extreme percentages (far from 50) mean stronger neutral stance
        distance_from_center = abs(sentiment_percentage - 50)
        index = min(len(primary_emojis) - 1, int(distance_from_center / 15))
    
    # Select emojis based on calculated index
    primary_emoji = primary_emojis[index]
    secondary_emoji = secondary_emojis[min(index, len(secondary_emojis) - 1)]
    
    return {"primary": primary_emoji, "secondary": secondary_emoji}

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
        # Initialize missing attributes if they don't exist
        # This ensures the model works even if these were missing when it was serialized
        
        # Add hashtags attributes if missing
        if not hasattr(sentiment_model, 'positive_hashtags'):
            sentiment_model.positive_hashtags = [
                "love", "happy", "blessed", "grateful", "joy", "beautiful", "amazing", 
                "awesome", "excited", "wonderful", "success", "inspiration", "goals",
                "win", "winning", "blessed", "gratitude", "positive", "positivity"
            ]
            print("Added missing positive_hashtags attribute to sentiment model")
            
        if not hasattr(sentiment_model, 'negative_hashtags'):
            sentiment_model.negative_hashtags = [
                "sad", "angry", "upset", "disappointed", "fail", "failure", "hate", 
                "depressed", "depression", "anxiety", "stressed", "tired", "exhausted",
                "worried", "heartbroken", "brokenheart", "lonely", "alone", "hurt"
            ]
            print("Added missing negative_hashtags attribute to sentiment model")
            
        # Add cache attributes if missing
        if not hasattr(sentiment_model, 'analysis_cache'):
            sentiment_model.analysis_cache = {}
            print("Added missing analysis_cache attribute to sentiment model")
            
        if not hasattr(sentiment_model, 'cache_max_size'):
            sentiment_model.cache_max_size = 1000
            print("Added missing cache_max_size attribute to sentiment model")
            
        # Add keyword attributes if missing
        if not hasattr(sentiment_model, 'mixed_keywords'):
            sentiment_model.mixed_keywords = [
                "bittersweet", "mixed feelings", "conflicted", "ambivalent"
            ]
            print("Added missing mixed_keywords attribute to sentiment model")
            
        # Add emoji-related attributes if missing
        if not hasattr(sentiment_model, 'positive_emojis'):
            sentiment_model.positive_emojis = ["😀", "😃", "😄", "😁", "😊", "👍"]
            print("Added missing positive_emojis attribute to sentiment model")
            
        if not hasattr(sentiment_model, 'negative_emojis'):
            sentiment_model.negative_emojis = ["😞", "😔", "😢", "😭", "👎"]
            print("Added missing negative_emojis attribute to sentiment model")
            
        if not hasattr(sentiment_model, 'neutral_emojis'):
            sentiment_model.neutral_emojis = ["😐", "😑", "😶"]
            print("Added missing neutral_emojis attribute to sentiment model")
            
        # Add context-sensitive attributes if missing
        if not hasattr(sentiment_model, 'context_sensitive_emojis'):
            sentiment_model.context_sensitive_emojis = {}
            print("Added missing context_sensitive_emojis attribute to sentiment model")
            
        # Add critical keywords attribute if missing
        if not hasattr(sentiment_model, 'critical_keywords'):
            sentiment_model.critical_keywords = [
                "constructive", "feedback", "suggest", "recommend", "improvement"
            ]
            print("Added missing critical_keywords attribute to sentiment model")
            
        # Override detect_critical_keywords method if it references 'kw'
        def safe_detect_critical_keywords(text):
            """Check if any critical/constructive keywords are in the text."""
            if hasattr(sentiment_model, 'critical_keywords'):
                return any(keyword in text for keyword in sentiment_model.critical_keywords)
            return False
            
        # Replace the original method with our safe version
        sentiment_model.detect_critical_keywords = safe_detect_critical_keywords
            
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

        # Get sentiment emojis based on category and percentage
        sentiment_emojis = get_sentiment_emojis(category_index, sentiment_percentage)

        # Add percentage to the label
        display_label = f"{sentiment_label} ({sentiment_percentage}%) {sentiment_emojis['primary']} {sentiment_emojis['secondary']}"

        # Prepare the response with detailed sentiment information
        response = {
            "text": text, 
            "prediction": int(category_index),  # Convert to int for JSON serialization
            "sentiment": display_label,
            "sentiment_percentage": sentiment_percentage,
            "sentiment_emojis": sentiment_emojis,
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
    """Generate detailed reasoning for why the text has the assigned sentiment."""
    # Convert sentiment category to int if it's a string
    if isinstance(sentiment_category, str):
        try:
            sentiment_category = int(sentiment_category)
        except ValueError:
            sentiment_category = 2  # Default to neutral
    
    # Improved reasoning templates with more variation and depth
    reasoning_templates = {
        0: [  # Overwhelmingly negative
            "The text contains extremely negative language with strong accusations or hostile rhetoric that suggests intense disapproval.",
            "Multiple negative expressions and emotionally charged language indicate severe criticism or anger.",
            "The content uses aggressive terminology that conveys profound dissatisfaction or outrage.",
            "Strong negative sentiment markers throughout the text indicate extreme opposition or distress."
        ],
        1: [  # Negative
            "The language contains critical terms and phrases that suggest general dissatisfaction.",
            "Several negative expressions are used to convey disagreement or disappointment.",
            "The text presents complaints or unfavorable assessments without extreme intensity.",
            "The content frames the subject in a negative light through subtle criticism."
        ],
        2: [  # Neutral
            "The text maintains a balanced perspective without strong emotional indicators.",
            "The content primarily conveys information in a factual manner without clear bias.",
            "There is an approximately equal balance of positive and negative elements, or absence of both.",
            "The statements are presented objectively with focus on reporting rather than evaluation."
        ],
        3: [  # Positive
            "The language incorporates supportive terminology and approving statements.",
            "Several positive expressions are used to convey satisfaction or agreement.",
            "The text frames subjects in a favorable light without excessive enthusiasm.",
            "The content shows optimism and constructive perspectives on the discussed topics."
        ],
        4: [  # Overwhelmingly positive
            "The text uses extremely enthusiastic language with strong expressions of admiration or joy.",
            "Multiple superlatives and celebratory phrases indicate exceptional approval or excitement.",
            "The content shows passionate support through consistent positive terminology.",
            "Strong positive sentiment markers throughout the text convey intense appreciation."
        ]
    }
    
    # Select reasoning template based on sentiment category
    if sentiment_category in reasoning_templates:
        templates = reasoning_templates[sentiment_category]
        import random
        reasoning = random.choice(templates)
    else:
        reasoning = "The text contains mixed or ambiguous sentiment signals."
    
    # Enhanced context detection for more specific analysis
    # Political content
    political_keywords = ['government', 'election', 'vote', 'democracy', 'political', 'policy', 'president', 
                         'minister', 'parliament', 'congress', 'senate', 'rights', 'freedom', 'law']
    
    # Conflict/war related
    conflict_keywords = ['war', 'attack', 'military', 'soldiers', 'troops', 'weapon', 'battle', 'fight', 
                        'conflict', 'violence', 'terrorist', 'terrorism', 'bombing', 'killed']
    
    # Economic content
    economic_keywords = ['economy', 'market', 'stock', 'price', 'inflation', 'economic', 'financial', 
                        'trade', 'business', 'company', 'investment', 'bank', 'money', 'dollar', 'rupee']
    
    # Social issues
    social_keywords = ['community', 'society', 'social', 'minority', 'discrimination', 'equality', 
                      'justice', 'reform', 'protest', 'movement', 'rights', 'gender', 'race', 'religion']
    
    # Add context-specific reasoning
    if any(keyword in text.lower() for keyword in political_keywords):
        if sentiment_category <= 1:  # Negative sentiments
            reasoning += " The text discusses political issues with a critical or concerned perspective, potentially highlighting governance problems or policy disagreements."
        elif sentiment_category >= 3:  # Positive sentiments
            reasoning += " The text discusses political topics with an optimistic or supportive tone, possibly endorsing certain policies or leadership actions."
        else:  # Neutral
            reasoning += " The text discusses political topics in a balanced manner, presenting information without strong partisan leaning."
    
    elif any(keyword in text.lower() for keyword in conflict_keywords):
        if sentiment_category <= 1:  # Negative sentiments
            reasoning += " The content references conflict or violence with a tone of concern, criticism, or alarm about the situation."
        elif sentiment_category >= 3:  # Positive sentiments
            reasoning += " Despite referencing conflict, the text maintains a hopeful or constructive perspective, possibly focusing on resolution or peace efforts."
        else:  # Neutral
            reasoning += " The text references conflict or security issues in a factual reporting style without emotional loading."
    
    elif any(keyword in text.lower() for keyword in economic_keywords):
        if sentiment_category <= 1:  # Negative sentiments
            reasoning += " The discussion of economic matters takes a pessimistic view, possibly highlighting problems or downturns."
        elif sentiment_category >= 3:  # Positive sentiments
            reasoning += " The economic content is framed positively, potentially highlighting growth, opportunity, or recovery."
        else:  # Neutral
            reasoning += " Economic information is presented in a factual manner, focusing on data rather than evaluation."
            
    elif any(keyword in text.lower() for keyword in social_keywords):
        if sentiment_category <= 1:  # Negative sentiments
            reasoning += " The discussion of social issues expresses concern or criticism about societal problems or inequalities."
        elif sentiment_category >= 3:  # Positive sentiments
            reasoning += " The social topics are discussed with optimism, possibly focusing on progress, solutions, or community strength."
        else:  # Neutral
            reasoning += " Social matters are presented with balanced perspective, acknowledging complexity without strong evaluative stance."
    
    return reasoning

def generate_model_opinion(sentiment_category, sentiment_label):
    """
    Generate a more nuanced opinion statement from the model based on the sentiment category and label.
    
    Args:
        sentiment_category: Integer category (0-4) or string representation
        sentiment_label: String label for the sentiment
        
    Returns:
        A contextual opinion about the content
    """
    # Try to get percentage from label if available
    percentage = None
    if sentiment_label and "(" in sentiment_label and "%" in sentiment_label:
        try:
            percentage = int(sentiment_label.split("(")[1].split("%")[0].strip())
        except (ValueError, IndexError):
            pass
    
    # Convert category to int if it's a string
    try:
        category = int(sentiment_category)
    except (ValueError, TypeError):
        category = 2  # Default to neutral
    
    # More nuanced opinions with intensity gradations based on percentage
    if percentage is not None:
        if category == 0:  # Overwhelmingly negative
            if percentage < 5:
                return "This content is extremely negative and contains hostile or alarming language."
            elif percentage < 15:
                return "This content is highly negative with strong expressions of criticism or disapproval."
            else:
                return "This content is very negative and may evoke strong negative emotions."
        elif category == 1:  # Negative
            if percentage < 25:
                return "This content leans negative with several critical or disapproving elements."
            else:
                return "This content is somewhat negative, expressing mild dissatisfaction or concern."
        elif category == 2:  # Neutral
            if percentage < 45:
                return "This content is mostly neutral with slight negative undertones."
            elif percentage > 55:
                return "This content is mostly neutral with slight positive undertones."
            else:
                return "This content is neutral and presents information without strong emotion."
        elif category == 3:  # Positive
            if percentage > 75:
                return "This content is quite positive, expressing clear satisfaction or approval."
            else:
                return "This content leans positive with supportive or optimistic elements."
        elif category == 4:  # Overwhelmingly positive
            if percentage > 95:
                return "This content is extremely positive with enthusiastic or celebratory language."
            elif percentage > 85:
                return "This content is highly positive with strong expressions of appreciation or joy."
            else:
                return "This content is very positive and conveys strong positive emotions."
    
    # Fallback to basic opinions if percentage is not available
    basic_opinions = {
        0: "This content is highly negative and may evoke strong negative emotions.",
        1: "This content is negative and expresses dissatisfaction or criticism.",
        2: "This content is neutral and presents information without strong emotion.",
        3: "This content is positive and expresses approval or satisfaction.",
        4: "This content is highly positive and conveys strong positive emotions."
    }
    
    return basic_opinions.get(category, "This content is neutral.")

@app.route("/api/sentiment", methods=["POST"])
def sentiment_analysis():
    try:
        data = request.get_json()
        if not validate_input(data):
            logging.warning("Invalid input received.")
            return jsonify({"error": "Invalid input."}), 400

        text = data.get("text", "")
        sentiment = analyze_sentiment(text)
        logging.info(f"Sentiment analysis successful for text: {text}")
        return jsonify(sentiment)

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" MOOD MAP API SERVER ".center(80, "="))
    print("=" * 80)
    print("\n📋 Request logging enabled - monitoring for extension requests")
    
    # Load models and configuration
    print("\n🔍 Checking sentiment model...")
    if sentiment_model:
        print("✅ Sentiment model loaded successfully!")
    else:
        print("❌ ERROR: Sentiment model could not be loaded. API will not work properly.")
    
    # Preload the summarization model at startup for faster first request
    print("\n🔍 Loading summarization model...")
    load_summarization_model()
    
    # Check if SSL certificate files exist
    cert_path = os.path.join(os.path.dirname(__file__), "cert.pem")
    key_path = os.path.join(os.path.dirname(__file__), "key.pem")
    
    print("\n🔍 Checking SSL certificates...")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        print("✅ Found SSL certificates, starting server with HTTPS")
        try:
            print("\n" + "*" * 80)
            print(" STARTING SERVER - HTTPS MODE ".center(80, "*"))
            print("*" * 80)
            print("\n🚀 Server starting at https://localhost:5000")
            print("📢 IMPORTANT: The server is now running! Press Ctrl+C to stop.")
            print("🌐 Access the API at: https://localhost:5000")
            print("\n" + "*" * 80 + "\n")
            app.run(ssl_context=(cert_path, key_path))
        except Exception as e:
            print(f"\n❌ ERROR: Failed to start server with HTTPS: {e}")
            print("⚠️ Attempting to start with HTTP instead...")
            try:
                print("\n" + "*" * 80)
                print(" STARTING SERVER - HTTP MODE (FALLBACK) ".center(80, "*"))
                print("*" * 80)
                print("\n🚀 Server starting at http://localhost:5000")
                print("📢 IMPORTANT: The server is now running! Press Ctrl+C to stop.")
                print("🌐 Access the API at: http://localhost:5000")
                print("\n" + "*" * 80 + "\n")
                app.run(host="0.0.0.0", port=5000)
            except Exception as e2:
                print(f"\n❌ CRITICAL ERROR: Could not start server: {e2}")
    else:
        print("⚠️ SSL certificates not found, starting server with HTTP only")
        try:
            print("\n" + "*" * 80)
            print(" STARTING SERVER - HTTP MODE ".center(80, "*"))
            print("*" * 80)
            print("\n🚀 Server starting at http://localhost:5000")
            print("📢 IMPORTANT: The server is now running! Press Ctrl+C to stop.")
            print("🌐 Access the API at: http://localhost:5000")
            print("\n" + "*" * 80 + "\n")
            app.run(host="0.0.0.0", port=5000)
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Could not start server: {e}")
            print("⛔ The server failed to start. Please check the errors above.")
            sys.exit(1)